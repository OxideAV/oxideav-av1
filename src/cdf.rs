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
//!   * **Motion-mode** (round 23):
//!       * `Default_Motion_Mode_Cdf` (`motion_mode`, keyed by
//!         `MiSize` — a straight `BLOCK_SIZES` index, no
//!         neighbour-context arithmetic).
//!
//!   * **Compound prediction** (round 24):
//!       * `Default_Comp_Group_Idx_Cdf` (`comp_group_idx`, binary,
//!         keyed by a precomputed `ctx ∈ 0..COMP_GROUP_IDX_CONTEXTS`).
//!       * `Default_Compound_Idx_Cdf` (`compound_idx`, binary, keyed
//!         by a precomputed `ctx ∈ 0..COMPOUND_IDX_CONTEXTS`).
//!       * `Default_Compound_Type_Cdf` (`compound_type`, keyed by
//!         `MiSize` — a straight `BLOCK_SIZES` index, no
//!         neighbour-context arithmetic).
//!
//!   * **Inter-frame intra mode** (round 134):
//!       * `Default_Y_Mode_Cdf` (`y_mode`, keyed by
//!         `ctx = Size_Group[ MiSize ] ∈ 0..BLOCK_SIZE_GROUPS`).
//!       * `Default_Uv_Mode_Cfl_Not_Allowed_Cdf` (`uv_mode` when
//!         chroma-from-luma is not allowed, keyed by `YMode`).
//!       * `Default_Uv_Mode_Cfl_Allowed_Cdf` (`uv_mode` when
//!         chroma-from-luma is allowed, keyed by `YMode`).
//!
//!   * **Angle-delta** (round 135):
//!       * `Default_Angle_Delta_Cdf` (`angle_delta_y` / `angle_delta_uv`,
//!         keyed by the directional intra mode rebased onto
//!         `0..DIRECTIONAL_MODES` via `YMode - V_PRED` /
//!         `UVMode - V_PRED`).
//!
//!   * **Coefficient-token entry sub-group** (round 136) — the
//!     `init_coeff_cdfs` gateway to tile-content decode:
//!       * `Default_Txb_Skip_Cdf` (`all_zero`, the transform-block skip
//!         flag).
//!       * `Default_Eob_Pt_{16,32,64,128,256,512,1024}_Cdf` (the
//!         end-of-block position class `eob_pt_*`).
//!       * `Default_Eob_Extra_Cdf` (the binary `eob_extra` flag).
//!       * `Default_Dc_Sign_Cdf` (the binary `dc_sign`).
//!
//! Unlike the non-coeff CDFs the coefficient-entry arrays are reset by
//! the separate [`TileCdfContext::init_coeff_cdfs`] (`base_q_idx` →
//! `idx` via [`coeff_cdf_q_ctx`]); the working copy drops the
//! `COEFF_CDF_Q_CTXS` axis, selecting `Default_*_Cdf[ idx ]`.
//!
//!   * **Intra-frame transform-type** (round 137):
//!       * `Default_Intra_Tx_Type_Set1_Cdf` (`intra_tx_type`,
//!         `TX_SET_INTRA_1`, 4x4 / 8x8 intra blocks; 7 symbols).
//!       * `Default_Intra_Tx_Type_Set2_Cdf` (`intra_tx_type`,
//!         `TX_SET_INTRA_2`, 4x4 / 8x8 / 16x16 intra blocks; 5 symbols).
//!
//!   * **Coefficient `coeff_base_eob` sub-group** (round 138) — the
//!     first of the `init_coeff_cdfs` `coeff_base` / `coeff_base_eob` /
//!     `coeff_br` braid:
//!       * `Default_Coeff_Base_Eob_Cdf` (`coeff_base_eob`, the base
//!         level of the last non-zero coefficient).
//!
//!   * **Coefficient `coeff_base` sub-group** (round 139) — the second
//!     of the `init_coeff_cdfs` `coeff_base` / `coeff_base_eob` /
//!     `coeff_br` braid:
//!       * `Default_Coeff_Base_Cdf` (`coeff_base`, the base level of
//!         each non-EOB coefficient; codes the 4-symbol alphabet
//!         `0..3`).
//!
//!   * **Coefficient `coeff_br` sub-group** (round 140) — the LAST of
//!     the `init_coeff_cdfs` `coeff_base` / `coeff_base_eob` /
//!     `coeff_br` braid:
//!       * `Default_Coeff_Br_Cdf` (`coeff_br`, the per-coefficient
//!         base-range increment used to push a level above
//!         `NUM_BASE_LEVELS`; codes the `BR_CDF_SIZE`-symbol alphabet
//!         `0..BR_CDF_SIZE`, stacked up to
//!         `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)` times per
//!         coefficient per §5.11.39).
//!
//! With this table all three coefficient-CDF braid members are landed;
//! the next gate is the §8.3.2 `get_coeff_base_ctx()` / `get_br_ctx()`
//! neighbour-derivation helpers (deferred — they need tile-content
//! walker state). The remaining `Default_*_Cdf` arrays of §9.4 (the
//! inter-intra group), and the §8.3.2 `split_or_horz` /
//! `split_or_vert` / … selections are a clear followup: each is a
//! mechanical transcription of one §9.4 table plus its §8.3.2
//! paragraph, slotted into the same [`TileCdfContext`] shape used here.
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

/// `TX_SET_INTRA_1` (§6.10.19) — the full-intra transform set
/// (`Tx_Type_Intra_Inv_Set1`, 7 symbols including `IDTX`, `DCT_DCT`,
/// `V_DCT`, `H_DCT`, `ADST_ADST`, `ADST_DCT`, `DCT_ADST`). Returned by
/// §5.11.48 `get_tx_set()` on the `is_inter == 0` branch when neither
/// `reduced_tx_set` nor `txSzSqr == TX_16X16` apply.
pub const TX_SET_INTRA_1: u32 = 1;

/// `TX_SET_INTRA_2` (§6.10.19) — the reduced-intra transform set
/// (`Tx_Type_Intra_Inv_Set2`, 5 symbols: `IDTX`, `DCT_DCT`, `ADST_ADST`,
/// `ADST_DCT`, `DCT_ADST`). Returned by §5.11.48 `get_tx_set()` on the
/// `is_inter == 0` branch when `reduced_tx_set == 1` or
/// `txSzSqr == TX_16X16`.
pub const TX_SET_INTRA_2: u32 = 2;

/// Symbol count for §6.10.19 `TX_SET_INTRA_1` (`Tx_Type_Intra_Inv_Set1`
/// has 7 entries). Drives the row width of
/// [`DEFAULT_INTRA_TX_TYPE_SET1_CDF`] (`TX_TYPES_INTRA_SET1 + 1`
/// cumulative frequencies + the §8.3 adaptation counter, matching the
/// `[8]` row width in the spec's §9.4 listing).
pub const TX_TYPES_INTRA_SET1: usize = 7;

/// Symbol count for §6.10.19 `TX_SET_INTRA_2` (`Tx_Type_Intra_Inv_Set2`
/// has 5 entries). Drives the row width of
/// [`DEFAULT_INTRA_TX_TYPE_SET2_CDF`] (`TX_TYPES_INTRA_SET2 + 1`
/// cumulative frequencies + the §8.3 adaptation counter, matching the
/// `[6]` row width in the spec's §9.4 listing).
pub const TX_TYPES_INTRA_SET2: usize = 5;

/// First-axis length of [`DEFAULT_INTRA_TX_TYPE_SET1_CDF`] — two entries
/// for `Tx_Size_Sqr[ txSz ] ∈ { TX_4X4, TX_8X8 }`, the only sizes that
/// reach `TX_SET_INTRA_1` per §5.11.48 (`txSzSqr == TX_16X16` is routed
/// to `TX_SET_INTRA_2`, `txSzSqrUp == TX_32X32` to `TX_SET_DCTONLY`).
pub const INTRA_TX_TYPE_SET1_SIZES: usize = 2;

/// First-axis length of [`DEFAULT_INTRA_TX_TYPE_SET2_CDF`] — three
/// entries for `Tx_Size_Sqr[ txSz ] ∈ { TX_4X4, TX_8X8, TX_16X16 }`, the
/// reachable sizes for the reduced intra-tx-type set (per §5.11.48:
/// `txSzSqrUp > TX_32X32` already returns `TX_SET_DCTONLY`, and only
/// `txSzSqr == TX_16X16` or `reduced_tx_set == 1` routes to
/// `TX_SET_INTRA_2`).
pub const INTRA_TX_TYPE_SET2_SIZES: usize = 3;

/// `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]` (§8.3.2
/// `intra_tx_type`) — when `use_filter_intra == 1`, the §8.3.2 `intraDir`
/// axis is `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]` rather
/// than `YMode`. The spec lists the entries as
/// `{ DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED }`; their numeric
/// values follow the §6.10.x intra-mode enumeration (`DC_PRED = 0`,
/// `V_PRED = 1`, `H_PRED = 2`, `D157_PRED = 6`).
pub const FILTER_INTRA_MODE_TO_INTRA_DIR: [usize; INTRA_FILTER_MODES] = [0, 1, 2, 6, 0];

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

/// `MOTION_MODES` per §3 — number of values for `motion_mode`
/// (`SIMPLE = 0`, `OBMC = 1`, `LOCALWARP = 2`; see §6.10.26 semantics).
/// Drives the row width of [`DEFAULT_MOTION_MODE_CDF`]
/// (`MOTION_MODES + 1` = three cumulative frequencies + the §8.3
/// adaptation counter).
pub const MOTION_MODES: usize = 3;

/// `COMPOUND_TYPES` per §3 — number of values for `compound_type`
/// (the §8.3.2 binary symbol selecting `COMPOUND_WEDGE` vs
/// `COMPOUND_DIFFWTD`; see §6.10.24 semantics). Drives the row width
/// of [`DEFAULT_COMPOUND_TYPE_CDF`] (`COMPOUND_TYPES + 1` = two
/// cumulative frequencies + the §8.3 adaptation counter).
pub const COMPOUND_TYPES: usize = 2;

/// `COMP_GROUP_IDX_CONTEXTS` per §3 — number of contexts for
/// `comp_group_idx`. Drives the outer dimension of
/// [`DEFAULT_COMP_GROUP_IDX_CDF`]. The §8.3.2 selection clamps the
/// computed context with `ctx = Min(5, ctx)`, so the valid range is
/// `0..COMP_GROUP_IDX_CONTEXTS`.
pub const COMP_GROUP_IDX_CONTEXTS: usize = 6;

/// `COMPOUND_IDX_CONTEXTS` per §3 — number of contexts for
/// `compound_idx`. Drives the outer dimension of
/// [`DEFAULT_COMPOUND_IDX_CDF`]; valid range is
/// `0..COMPOUND_IDX_CONTEXTS`.
pub const COMPOUND_IDX_CONTEXTS: usize = 6;

/// `BLOCK_SIZE_GROUPS` (§3) — number of contexts when decoding `y_mode`
/// (the inter-frame, non-keyframe luma intra-mode element). Drives the
/// outer dimension of [`DEFAULT_Y_MODE_CDF`]. The §8.3.2 selection
/// computes `ctx = Size_Group[ MiSize ]`, in `0..BLOCK_SIZE_GROUPS`.
pub const BLOCK_SIZE_GROUPS: usize = 4;

/// `UV_INTRA_MODES_CFL_NOT_ALLOWED` (§3) — number of values for `uv_mode`
/// when chroma-from-luma is not allowed (i.e. `UV_CFL_PRED` is excluded).
/// Drives the row width of [`DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF`]
/// (`UV_INTRA_MODES_CFL_NOT_ALLOWED + 1` cumulative frequencies + the
/// §8.3 adaptation counter).
pub const UV_INTRA_MODES_CFL_NOT_ALLOWED: usize = 13;

/// `UV_INTRA_MODES_CFL_ALLOWED` (§3) — number of values for `uv_mode`
/// when chroma-from-luma is allowed (i.e. `UV_CFL_PRED` is included).
/// Drives the row width of [`DEFAULT_UV_MODE_CFL_ALLOWED_CDF`]
/// (`UV_INTRA_MODES_CFL_ALLOWED + 1` cumulative frequencies + the §8.3
/// adaptation counter).
pub const UV_INTRA_MODES_CFL_ALLOWED: usize = 14;

/// `Size_Group[ BLOCK_SIZES ]` (§8.3.2) — maps a luma block size
/// (`MiSize`) into the `y_mode` context (and, after `- 1`, several other
/// intra-syntax contexts). Used by the §8.3.2 `y_mode` selection as
/// `ctx = Size_Group[ MiSize ]`, yielding a value in
/// `0..BLOCK_SIZE_GROUPS`.
pub const SIZE_GROUP: [usize; BLOCK_SIZES] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
];

/// `DIRECTIONAL_MODES` (§3) — number of directional intra modes, i.e. the
/// span of `YMode` / `UVMode` values from `V_PRED` to `D67_PRED`
/// inclusive. Drives the outer dimension of [`DEFAULT_ANGLE_DELTA_CDF`].
/// The §8.3.2 `angle_delta_y` / `angle_delta_uv` selections index that
/// table by `YMode - V_PRED` / `UVMode - V_PRED`, a value in
/// `0..DIRECTIONAL_MODES`.
pub const DIRECTIONAL_MODES: usize = 8;

/// `MAX_ANGLE_DELTA` (§3) — maximum magnitude of `AngleDeltaY` and
/// `AngleDeltaUV`. The coded `angle_delta_y` / `angle_delta_uv` syntax
/// elements are biased by `MAX_ANGLE_DELTA` so as to encode a positive
/// value, so they range over `0..(2 * MAX_ANGLE_DELTA + 1)`; that span
/// (`2 * MAX_ANGLE_DELTA + 1 == 7`) is the cumulative-frequency width of
/// [`DEFAULT_ANGLE_DELTA_CDF`].
pub const MAX_ANGLE_DELTA: usize = 3;

/// `V_PRED` (§6.10.x intra-mode enumeration) — the first directional
/// intra mode (`DC_PRED == 0`, `V_PRED == 1`, …, `D67_PRED == 8`). The
/// §8.3.2 `angle_delta` selections rebase a directional `YMode` / `UVMode`
/// onto `0..DIRECTIONAL_MODES` by subtracting this value.
pub const V_PRED: usize = 1;

/// `INTERINTRA_MODES` (§3 / §6.10.27) — number of values for
/// `interintra_mode` (`II_DC_PRED = 0`, `II_V_PRED = 1`, `II_H_PRED = 2`,
/// `II_SMOOTH_PRED = 3`). Drives the row width of
/// [`DEFAULT_INTER_INTRA_MODE_CDF`] (`INTERINTRA_MODES + 1` cumulative
/// frequencies + the §8.3 adaptation counter). The §5.11.28
/// `read_interintra_mode` syntax gate restricts coded blocks to the
/// `BLOCK_8X8`..`BLOCK_32X32` band (per §6.10.27), where `Size_Group[
/// MiSize ]` evaluates to `1`, `2`, or `3` — hence the `[
/// BLOCK_SIZE_GROUPS - 1 ]` outer dimension on the matching default
/// CDFs.
pub const INTERINTRA_MODES: usize = 4;

/// `WEDGE_TYPES` (§3) — number of directions for the wedge mask process
/// (the spec text reads: *"Number of directions for the wedge mask
/// process"*). Drives the symbol axis of [`DEFAULT_WEDGE_INDEX_CDF`]
/// (`WEDGE_TYPES + 1` cumulative frequencies + the §8.3 adaptation
/// counter). `wedge_index` is read by the §5.11.28
/// `read_interintra_mode` (the inter-intra wedge sub-branch, when
/// `wedge_interintra == 1`) and by the §5.11.29 `read_compound_type`
/// (the inter-inter `COMPOUND_WEDGE` branch, when
/// `compound_type == COMPOUND_WEDGE`) syntax elements; the same default
/// CDF is selected by both call sites per §8.3.2 (the cdf is
/// `TileWedgeIndexCdf[ MiSize ]`).
pub const WEDGE_TYPES: usize = 16;

// ---------------------------------------------------------------------
// Round 136 — coefficient-token entry constants (§3). These drive the
// outer/inner dimensions of the `init_coeff_cdfs` entry sub-group:
// `Default_Txb_Skip_Cdf` + `Default_Eob_Pt_*_Cdf` + `Default_Eob_Extra_Cdf`
// + `Default_Dc_Sign_Cdf` (§9.4). `PLANE_TYPES` / `TX_SIZES` are shared
// with later coeff_base/coeff_br tables (deferred to a later round).
// ---------------------------------------------------------------------

/// `PLANE_TYPES` (§3) — number of different plane types (luma or chroma).
/// Drives the per-plane axis of the EOB-position / DC-sign coefficient
/// CDFs (`Default_Eob_Pt_*_Cdf`, `Default_Eob_Extra_Cdf`,
/// `Default_Dc_Sign_Cdf`).
pub const PLANE_TYPES: usize = 2;

/// `COEFF_CDF_Q_CTXS` (§3) — number of selectable context types for the
/// `coeff( )` syntax structure. This is the outermost dimension of every
/// `init_coeff_cdfs` default table; the working `Tile*Cdf` copy drops it,
/// selecting `Default_*_Cdf[ idx ]` where `idx` is derived from
/// `base_q_idx` (§8.3.1 `init_coeff_cdfs`). See [`coeff_cdf_q_ctx`].
pub const COEFF_CDF_Q_CTXS: usize = 4;

/// `TXB_SKIP_CONTEXTS` (§3) — number of contexts for `all_zero` (the
/// transform-block skip flag). Indexes the inner axis of
/// [`DEFAULT_TXB_SKIP_CDF`].
pub const TXB_SKIP_CONTEXTS: usize = 13;

/// `EOB_COEF_CONTEXTS` (§3) — number of contexts for `eob_extra`. Indexes
/// the innermost context axis of [`DEFAULT_EOB_EXTRA_CDF`].
pub const EOB_COEF_CONTEXTS: usize = 9;

/// `DC_SIGN_CONTEXTS` (§3) — number of contexts for `dc_sign`. Indexes the
/// inner axis of [`DEFAULT_DC_SIGN_CDF`].
pub const DC_SIGN_CONTEXTS: usize = 3;

/// `SIG_COEF_CONTEXTS_EOB` (§3) — number of contexts for `coeff_base_eob`.
/// Indexes the innermost context axis of [`DEFAULT_COEFF_BASE_EOB_CDF`].
/// The §8.3.2 derivation reduces the full `get_coeff_base_ctx()` result
/// onto `0..SIG_COEF_CONTEXTS_EOB` via
/// `ctx = get_coeff_base_ctx(...) - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`.
pub const SIG_COEF_CONTEXTS_EOB: usize = 4;

/// `SIG_COEF_CONTEXTS` (§3) — number of contexts for `coeff_base`. Indexes
/// the innermost context axis of [`DEFAULT_COEFF_BASE_CDF`]. The §8.3.2
/// `get_coeff_base_ctx()` derivation returns a value in
/// `0..SIG_COEF_CONTEXTS` for non-EOB positions (the EOB position is
/// reduced onto `0..SIG_COEF_CONTEXTS_EOB` per the formula above). The
/// §3 partition tags `SIG_COEF_CONTEXTS_2D` (`= 26`) split this range
/// between the two-dimensional scan prefix and the one-dimensional
/// horizontal- / vertical-only tails.
pub const SIG_COEF_CONTEXTS: usize = 42;

/// `LEVEL_CONTEXTS` (§3) — number of contexts for `coeff_br`. Indexes the
/// innermost context axis of [`DEFAULT_COEFF_BR_CDF`]. The §8.3.2
/// `get_br_ctx()` derivation returns a value in `0..LEVEL_CONTEXTS` for
/// every transform-coefficient position that needs a base-range increment.
pub const LEVEL_CONTEXTS: usize = 21;

/// `BR_CDF_SIZE` (§3) — number of values for `coeff_br` (the alphabet
/// size of the per-coefficient base-range increment). Each `coeff_br`
/// read codes a value in `0..BR_CDF_SIZE`; up to
/// `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)` `coeff_br` reads can stack per
/// coefficient, as enumerated by §5.11.39. The CDF row therefore has
/// `BR_CDF_SIZE + 1` entries (`BR_CDF_SIZE` cumulative frequencies +
/// the §8.3 adaptation counter).
pub const BR_CDF_SIZE: usize = 4;

/// `SIG_COEF_CONTEXTS_2D` (§3) — the split point inside the
/// `0..SIG_COEF_CONTEXTS` `coeff_base` context range. The §8.3.2
/// `get_coeff_base_ctx()` two-dimensional branch returns values in
/// `0..SIG_COEF_CONTEXTS_2D`; the one-dimensional vertical- /
/// horizontal-only branch returns `SIG_COEF_CONTEXTS_2D` and above via
/// [`COEFF_BASE_POS_CTX_OFFSET`].
pub const SIG_COEF_CONTEXTS_2D: usize = 26;

/// `SIG_REF_DIFF_OFFSET_NUM` (§3) — the number of neighbour offsets
/// scanned by the §8.3.2 `get_coeff_base_ctx()` magnitude accumulation
/// (the row count of each [`SIG_REF_DIFF_OFFSET`] sub-table).
pub const SIG_REF_DIFF_OFFSET_NUM: usize = 5;

/// `NUM_BASE_LEVELS` (§3) — the number of base levels coded by
/// `coeff_base` before `coeff_br` increments take over. Used by the
/// §8.3.2 `get_br_ctx()` magnitude clamp
/// `Min(Quant[..], COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1)`.
pub const NUM_BASE_LEVELS: usize = 2;

/// `COEFF_BASE_RANGE` (§3) — the maximum cumulative base-range increment
/// a coefficient can accrue from `coeff_br` reads. Used by the §8.3.2
/// `get_br_ctx()` magnitude clamp (see [`NUM_BASE_LEVELS`]).
pub const COEFF_BASE_RANGE: usize = 12;

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
// Round 137 — intra-frame transform-type group (§9.4):
// `Default_Intra_Tx_Type_Set{1,2}_Cdf`. Codes `intra_tx_type` per
// §5.11.47 / §8.3.2 against the §6.10.19 intra transform set returned
// by §5.11.48 `get_tx_set()` on the `is_inter == 0` branch. Both
// tables carry an extra `intraDir` axis (`INTRA_MODES = 13`) on top of
// the `Tx_Size_Sqr` axis seen in the inter counterparts; `intraDir` is
// `YMode` when `use_filter_intra == 0` and
// `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]` otherwise (see
// [`FILTER_INTRA_MODE_TO_INTRA_DIR`]).
// ---------------------------------------------------------------------

/// `Default_Intra_Tx_Type_Set1_Cdf[ 2 ][ INTRA_MODES ][ 8 ]` (§9.4).
///
/// Selected by §8.3.2 when `set == TX_SET_INTRA_1` (the full intra
/// transform set, `Tx_Type_Intra_Inv_Set1` with 7 entries; the row
/// width is `TX_TYPES_INTRA_SET1 + 1 == 8` per the §8.3 adaptation-
/// counter convention). The first axis is `Tx_Size_Sqr[ txSz ] ∈
/// { TX_4X4, TX_8X8 }` — `txSzSqr == TX_16X16` is routed to
/// `TX_SET_INTRA_2` by §5.11.48 and never reaches this table. The
/// second axis is `intraDir ∈ 0..INTRA_MODES` (`YMode` or
/// `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]`).
pub const DEFAULT_INTRA_TX_TYPE_SET1_CDF: [[[u16; TX_TYPES_INTRA_SET1 + 1]; INTRA_MODES];
    INTRA_TX_TYPE_SET1_SIZES] = [
    [
        [1535, 8035, 9461, 12751, 23467, 27825, 32768, 0],
        [564, 3335, 9709, 10870, 18143, 28094, 32768, 0],
        [672, 3247, 3676, 11982, 19415, 23127, 32768, 0],
        [5279, 13885, 15487, 18044, 23527, 30252, 32768, 0],
        [4423, 6074, 7985, 10416, 25693, 29298, 32768, 0],
        [1486, 4241, 9460, 10662, 16456, 27694, 32768, 0],
        [439, 2838, 3522, 6737, 18058, 23754, 32768, 0],
        [1190, 4233, 4855, 11670, 20281, 24377, 32768, 0],
        [1045, 4312, 8647, 10159, 18644, 29335, 32768, 0],
        [202, 3734, 4747, 7298, 17127, 24016, 32768, 0],
        [447, 4312, 6819, 8884, 16010, 23858, 32768, 0],
        [277, 4369, 5255, 8905, 16465, 22271, 32768, 0],
        [3409, 5436, 10599, 15599, 19687, 24040, 32768, 0],
    ],
    [
        [1870, 13742, 14530, 16498, 23770, 27698, 32768, 0],
        [326, 8796, 14632, 15079, 19272, 27486, 32768, 0],
        [484, 7576, 7712, 14443, 19159, 22591, 32768, 0],
        [1126, 15340, 15895, 17023, 20896, 30279, 32768, 0],
        [655, 4854, 5249, 5913, 22099, 27138, 32768, 0],
        [1299, 6458, 8885, 9290, 14851, 25497, 32768, 0],
        [311, 5295, 5552, 6885, 16107, 22672, 32768, 0],
        [883, 8059, 8270, 11258, 17289, 21549, 32768, 0],
        [741, 7580, 9318, 10345, 16688, 29046, 32768, 0],
        [110, 7406, 7915, 9195, 16041, 23329, 32768, 0],
        [363, 7974, 9357, 10673, 15629, 24474, 32768, 0],
        [153, 7647, 8112, 9936, 15307, 19996, 32768, 0],
        [3511, 6332, 11165, 15335, 19323, 23594, 32768, 0],
    ],
];

/// `Default_Intra_Tx_Type_Set2_Cdf[ 3 ][ INTRA_MODES ][ 6 ]` (§9.4).
///
/// Selected by §8.3.2 when `set == TX_SET_INTRA_2` (the reduced intra
/// transform set, `Tx_Type_Intra_Inv_Set2` with 5 entries; the row
/// width is `TX_TYPES_INTRA_SET2 + 1 == 6` per the §8.3 adaptation-
/// counter convention). The first axis is `Tx_Size_Sqr[ txSz ] ∈
/// { TX_4X4, TX_8X8, TX_16X16 }` — `txSzSqrUp > TX_32X32` would
/// already have been routed to `TX_SET_DCTONLY` per §5.11.48. The
/// second axis is `intraDir ∈ 0..INTRA_MODES`.
///
/// Note: the §9.4 source lists the first two `Tx_Size_Sqr` rows
/// (`TX_4X4`, `TX_8X8`) as the uniform `{ 6554, 13107, 19661, 26214,
/// 32768, 0 }` distribution (i.e. a flat 5-way prior, since `32768 / 5
/// ≈ 6554`). Only the `TX_16X16` row carries adapted seeds. This
/// matches the §5.11.48 routing where `txSzSqr == TX_16X16` always
/// reaches `TX_SET_INTRA_2`, while the smaller sizes can only reach it
/// via `reduced_tx_set == 1`.
pub const DEFAULT_INTRA_TX_TYPE_SET2_CDF: [[[u16; TX_TYPES_INTRA_SET2 + 1]; INTRA_MODES];
    INTRA_TX_TYPE_SET2_SIZES] = [
    [
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
    ],
    [
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
        [6554, 13107, 19661, 26214, 32768, 0],
    ],
    [
        [1127, 12814, 22772, 27483, 32768, 0],
        [145, 6761, 11980, 26667, 32768, 0],
        [362, 5887, 11678, 16725, 32768, 0],
        [385, 15213, 18587, 30693, 32768, 0],
        [25, 2914, 23134, 27903, 32768, 0],
        [60, 4470, 11749, 23991, 32768, 0],
        [37, 3332, 14511, 21448, 32768, 0],
        [157, 6320, 13036, 17439, 32768, 0],
        [119, 6719, 12906, 29396, 32768, 0],
        [47, 5537, 12576, 21499, 32768, 0],
        [269, 6076, 11258, 23115, 32768, 0],
        [83, 5615, 12001, 17228, 32768, 0],
        [1968, 5556, 12023, 18547, 32768, 0],
    ],
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
// Round 23 — motion-mode group (§9.4): `Default_Motion_Mode_Cdf`. The
// §8.3.2 selection (§7.4.x "motion_mode: the cdf is given by
// `TileMotionModeCdf[ MiSize ]`") is a straight `BLOCK_SIZES` index —
// no neighbour-context arithmetic. Per the §9.4 note, first-dimension
// indices 0..=2 and 16..=17 are never reached by the §5.11.x
// `read_motion_mode` selection (those `MiSize` values do not satisfy
// the §5.11.x `Block_Width >= 8 && Block_Height >= 8` gate that lets
// the syntax element appear), but the table is still transcribed
// full-width.
// ---------------------------------------------------------------------

/// `Default_Motion_Mode_Cdf[ BLOCK_SIZES ][ MOTION_MODES + 1 ]` (§9.4).
/// Indexed by `MiSize` per the §8.3.2 selection; the innermost row codes
/// the `motion_mode ∈ { SIMPLE, OBMC, LOCALWARP }` symbol (`MOTION_MODES
/// = 3` cumulative frequencies plus the §8.3 adaptation counter that
/// starts at 0). Rows 0..=2 and 16..=17 are spec-flagged as unreachable
/// — see the §9.4 note quoted in the group banner above.
pub const DEFAULT_MOTION_MODE_CDF: [[u16; MOTION_MODES + 1]; BLOCK_SIZES] = [
    [10923, 21845, 32768, 0],
    [10923, 21845, 32768, 0],
    [10923, 21845, 32768, 0],
    [7651, 24760, 32768, 0],
    [4738, 24765, 32768, 0],
    [5391, 25528, 32768, 0],
    [19419, 26810, 32768, 0],
    [5123, 23606, 32768, 0],
    [11606, 24308, 32768, 0],
    [26260, 29116, 32768, 0],
    [20360, 28062, 32768, 0],
    [21679, 26830, 32768, 0],
    [29516, 30701, 32768, 0],
    [28898, 30397, 32768, 0],
    [30878, 31335, 32768, 0],
    [32507, 32558, 32768, 0],
    [10923, 21845, 32768, 0],
    [10923, 21845, 32768, 0],
    [28799, 31390, 32768, 0],
    [26431, 30774, 32768, 0],
    [28973, 31594, 32768, 0],
    [29742, 31203, 32768, 0],
];

// ---------------------------------------------------------------------
// Round 24 — compound-prediction group (§9.4):
// `Default_Comp_Group_Idx_Cdf`, `Default_Compound_Idx_Cdf`,
// `Default_Compound_Type_Cdf`. The first two are binary, keyed by a
// precomputed `ctx` (the §8.3.2 paragraphs derive `ctx` from
// neighbour state — `CompGroupIdxs` / `CompoundIdxs`, `AvailU` /
// `AvailL`, `AboveSingle` / `LeftSingle`, `AltRef` reference frames —
// which belongs in the future tile-walk; the selectors here take the
// already-computed index). `Default_Compound_Type_Cdf` is keyed by a
// straight `MiSize` index (the §8.3.2 selection text reads
// "`TileCompoundTypeCdf[ MiSize ]`"). Per the §9.4 note, first-
// dimension indices 0..=2, 10..=17 and 20..=21 of
// `Default_Compound_Type_Cdf` are never used (those block sizes don't
// satisfy the §5.11.x masked-compound gate); the table is still
// transcribed full-width.
// ---------------------------------------------------------------------

/// `Default_Compound_Idx_Cdf[ COMPOUND_IDX_CONTEXTS ][ 3 ]` (§9.4).
/// Binary symbol `compound_idx` (distance-weighted vs averaging blend;
/// see §6.10.24). Indexed by the precomputed `ctx` from the §8.3.2
/// `compound_idx` paragraph (`0..COMPOUND_IDX_CONTEXTS`). Each row is
/// two cumulative frequencies plus the §8.3 adaptation counter (0).
pub const DEFAULT_COMPOUND_IDX_CDF: [[u16; 3]; COMPOUND_IDX_CONTEXTS] = [
    [18244, 32768, 0],
    [12865, 32768, 0],
    [7053, 32768, 0],
    [13259, 32768, 0],
    [9334, 32768, 0],
    [4644, 32768, 0],
];

/// `Default_Comp_Group_Idx_Cdf[ COMP_GROUP_IDX_CONTEXTS ][ 3 ]` (§9.4).
/// Binary symbol `comp_group_idx` (whether `compound_idx` is present;
/// see §6.10.24). Indexed by the precomputed `ctx` from the §8.3.2
/// `comp_group_idx` paragraph (`ctx = Min(5, ctx)`, hence
/// `0..COMP_GROUP_IDX_CONTEXTS`). Each row is two cumulative
/// frequencies plus the §8.3 adaptation counter (0).
pub const DEFAULT_COMP_GROUP_IDX_CDF: [[u16; 3]; COMP_GROUP_IDX_CONTEXTS] = [
    [26607, 32768, 0],
    [22891, 32768, 0],
    [18840, 32768, 0],
    [24594, 32768, 0],
    [19934, 32768, 0],
    [22674, 32768, 0],
];

/// `Default_Compound_Type_Cdf[ BLOCK_SIZES ][ COMPOUND_TYPES + 1 ]`
/// (§9.4). Indexed by `MiSize` per the §8.3.2 selection; the innermost
/// row codes the binary `compound_type ∈ { COMPOUND_WEDGE,
/// COMPOUND_DIFFWTD }` symbol (`COMPOUND_TYPES = 2` cumulative
/// frequencies plus the §8.3 adaptation counter, which starts at 0).
/// Per the §9.4 note, rows 0..=2, 10..=17 and 20..=21 are never used in
/// the first dimension — see the group banner above.
pub const DEFAULT_COMPOUND_TYPE_CDF: [[u16; COMPOUND_TYPES + 1]; BLOCK_SIZES] = [
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [23431, 32768, 0],
    [13171, 32768, 0],
    [11470, 32768, 0],
    [9770, 32768, 0],
    [9100, 32768, 0],
    [8233, 32768, 0],
    [6172, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [11820, 32768, 0],
    [7701, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
];

/// `Default_Y_Mode_Cdf[ BLOCK_SIZE_GROUPS ][ INTRA_MODES + 1 ]` (§9.4).
/// The inter-frame (non-keyframe) luma intra-mode CDF, indexed by
/// `ctx = Size_Group[ MiSize ]` per the §8.3.2 `y_mode` selection. Each
/// row carries `INTRA_MODES = 13` cumulative frequencies (the last being
/// `1 << 15 == 32768`) plus the §8.3 adaptation counter (starts at 0).
/// Distinct from [`DEFAULT_INTRA_FRAME_Y_MODE_CDF`], which is the
/// keyframe / intra-frame variant keyed by `[abovemode][leftmode]`.
pub const DEFAULT_Y_MODE_CDF: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS] = [
    [
        22801, 23489, 24293, 24756, 25601, 26123, 26606, 27418, 27945, 29228, 29685, 30349, 32768,
        0,
    ],
    [
        18673, 19845, 22631, 23318, 23950, 24649, 25527, 27364, 28152, 29701, 29984, 30852, 32768,
        0,
    ],
    [
        19770, 20979, 23396, 23939, 24241, 24654, 25136, 27073, 27830, 29360, 29730, 30659, 32768,
        0,
    ],
    [
        20155, 21301, 22838, 23178, 23261, 23533, 23703, 24804, 25352, 26575, 27016, 28049, 32768,
        0,
    ],
];

/// `Default_Uv_Mode_Cfl_Not_Allowed_Cdf[ INTRA_MODES ][ UV_INTRA_MODES_CFL_NOT_ALLOWED + 1 ]`
/// (§9.4). The chroma intra-mode CDF used when chroma-from-luma is **not**
/// allowed; indexed by `YMode` per the §8.3.2 `uv_mode` selection. Each
/// row carries `UV_INTRA_MODES_CFL_NOT_ALLOWED = 13` cumulative
/// frequencies plus the §8.3 adaptation counter (starts at 0).
pub const DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF: [[u16; UV_INTRA_MODES_CFL_NOT_ALLOWED + 1];
    INTRA_MODES] = [
    [
        22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923, 28244, 30059, 30941, 31961, 32768,
        0,
    ],
    [
        9513, 26881, 26973, 27046, 27118, 27664, 27739, 27824, 28359, 29505, 29800, 31796, 32768, 0,
    ],
    [
        9845, 9915, 28663, 28704, 28757, 28780, 29198, 29822, 29854, 30764, 31777, 32029, 32768, 0,
    ],
    [
        13639, 13897, 14171, 25331, 25606, 25727, 25953, 27148, 28577, 30612, 31355, 32493, 32768,
        0,
    ],
    [
        9764, 9835, 9930, 9954, 25386, 27053, 27958, 28148, 28243, 31101, 31744, 32363, 32768, 0,
    ],
    [
        11825, 13589, 13677, 13720, 15048, 29213, 29301, 29458, 29711, 31161, 31441, 32550, 32768,
        0,
    ],
    [
        14175, 14399, 16608, 16821, 17718, 17775, 28551, 30200, 30245, 31837, 32342, 32667, 32768,
        0,
    ],
    [
        12885, 13038, 14978, 15590, 15673, 15748, 16176, 29128, 29267, 30643, 31961, 32461, 32768,
        0,
    ],
    [
        12026, 13661, 13874, 15305, 15490, 15726, 15995, 16273, 28443, 30388, 30767, 32416, 32768,
        0,
    ],
    [
        19052, 19840, 20579, 20916, 21150, 21467, 21885, 22719, 23174, 28861, 30379, 32175, 32768,
        0,
    ],
    [
        18627, 19649, 20974, 21219, 21492, 21816, 22199, 23119, 23527, 27053, 31397, 32148, 32768,
        0,
    ],
    [
        17026, 19004, 19997, 20339, 20586, 21103, 21349, 21907, 22482, 25896, 26541, 31819, 32768,
        0,
    ],
    [
        12124, 13759, 14959, 14992, 15007, 15051, 15078, 15166, 15255, 15753, 16039, 16606, 32768,
        0,
    ],
];

/// `Default_Uv_Mode_Cfl_Allowed_Cdf[ INTRA_MODES ][ UV_INTRA_MODES_CFL_ALLOWED + 1 ]`
/// (§9.4). The chroma intra-mode CDF used when chroma-from-luma **is**
/// allowed (so `UV_CFL_PRED` is a coded value); indexed by `YMode` per
/// the §8.3.2 `uv_mode` selection. Each row carries
/// `UV_INTRA_MODES_CFL_ALLOWED = 14` cumulative frequencies plus the §8.3
/// adaptation counter (starts at 0).
pub const DEFAULT_UV_MODE_CFL_ALLOWED_CDF: [[u16; UV_INTRA_MODES_CFL_ALLOWED + 1]; INTRA_MODES] = [
    [
        10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656, 15986, 20086, 20995, 22455, 24212,
        32768, 0,
    ],
    [
        4532, 19780, 20057, 20215, 20428, 21071, 21199, 21451, 22099, 24228, 24693, 27032, 29472,
        32768, 0,
    ],
    [
        5273, 5379, 20177, 20270, 20385, 20439, 20949, 21695, 21774, 23138, 24256, 24703, 26679,
        32768, 0,
    ],
    [
        6740, 7167, 7662, 14152, 14536, 14785, 15034, 16741, 18371, 21520, 22206, 23389, 24182,
        32768, 0,
    ],
    [
        4987, 5368, 5928, 6068, 19114, 20315, 21857, 22253, 22411, 24911, 25380, 26027, 26376,
        32768, 0,
    ],
    [
        5370, 6889, 7247, 7393, 9498, 21114, 21402, 21753, 21981, 24780, 25386, 26517, 27176,
        32768, 0,
    ],
    [
        4816, 4961, 7204, 7326, 8765, 8930, 20169, 20682, 20803, 23188, 23763, 24455, 24940, 32768,
        0,
    ],
    [
        6608, 6740, 8529, 9049, 9257, 9356, 9735, 18827, 19059, 22336, 23204, 23964, 24793, 32768,
        0,
    ],
    [
        5998, 7419, 7781, 8933, 9255, 9549, 9753, 10417, 18898, 22494, 23139, 24764, 25989, 32768,
        0,
    ],
    [
        10660, 11298, 12550, 12957, 13322, 13624, 14040, 15004, 15534, 20714, 21789, 23443, 24861,
        32768, 0,
    ],
    [
        10522, 11530, 12552, 12963, 13378, 13779, 14245, 15235, 15902, 20102, 22696, 23774, 25838,
        32768, 0,
    ],
    [
        10099, 10691, 12639, 13049, 13386, 13665, 14125, 15163, 15636, 19676, 20474, 23519, 25208,
        32768, 0,
    ],
    [
        3144, 5087, 7382, 7504, 7593, 7690, 7801, 8064, 8232, 9248, 9875, 10521, 29048, 32768, 0,
    ],
];

/// `Default_Angle_Delta_Cdf[ DIRECTIONAL_MODES ][ (2 * MAX_ANGLE_DELTA + 1) + 1 ]`
/// (§9.4). The CDF for the `angle_delta_y` / `angle_delta_uv` syntax
/// elements (the directional intra-prediction angle offset, biased by
/// `MAX_ANGLE_DELTA`). Indexed by the directional mode rebased onto
/// `0..DIRECTIONAL_MODES` (`YMode - V_PRED` for luma, `UVMode - V_PRED`
/// for chroma) per the §8.3.2 selection. Each row carries
/// `2 * MAX_ANGLE_DELTA + 1 = 7` cumulative frequencies (the last being
/// `1 << 15 == 32768`) plus the §8.3 adaptation counter (starts at 0).
pub const DEFAULT_ANGLE_DELTA_CDF: [[u16; (2 * MAX_ANGLE_DELTA + 1) + 1]; DIRECTIONAL_MODES] = [
    [2180, 5032, 7567, 22776, 26989, 30217, 32768, 0],
    [2301, 5608, 8801, 23487, 26974, 30330, 32768, 0],
    [3780, 11018, 13699, 19354, 23083, 31286, 32768, 0],
    [4581, 11226, 15147, 17138, 21834, 28397, 32768, 0],
    [1737, 10927, 14509, 19588, 22745, 28823, 32768, 0],
    [2664, 10176, 12485, 17650, 21600, 30495, 32768, 0],
    [2240, 11096, 15453, 20341, 22561, 28917, 32768, 0],
    [3605, 10428, 12459, 17676, 21244, 30655, 32768, 0],
];

// ---------------------------------------------------------------------
// Round 136 — coefficient-token entry sub-group default CDFs (§9.4),
// the gateway to tile-content decode. These are the tables reset by the
// §8.3.1 init_coeff_cdfs function: the transform-block skip flag
// (Default_Txb_Skip_Cdf), the end-of-block position class
// (Default_Eob_Pt_{16,32,64,128,256,512,1024}_Cdf), the EOB extra-bit
// (Default_Eob_Extra_Cdf), and the DC sign (Default_Dc_Sign_Cdf). The
// coeff_base / coeff_base_eob / coeff_br braid is deferred to a later
// round. Transcribed verbatim from the §9.4 source; the outer
// COEFF_CDF_Q_CTXS axis is selected by base_q_idx at init_coeff_cdfs.
// ---------------------------------------------------------------------

/// `Default_Txb_Skip_Cdf[ COEFF_CDF_Q_CTXS ][ TX_SIZES ][ TXB_SKIP_CONTEXTS ][ 3 ]`
/// (§9.4). Codes the `all_zero` (transform-block skip) flag. Selected at
/// `init_coeff_cdfs` by `idx` (the [`coeff_cdf_q_ctx`] value), then by
/// `txSzCtx` and the `all_zero` context.
pub const DEFAULT_TXB_SKIP_CDF: [[[[u16; 3]; TXB_SKIP_CONTEXTS]; TX_SIZES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [31849, 32768, 0],
            [5892, 32768, 0],
            [12112, 32768, 0],
            [21935, 32768, 0],
            [20289, 32768, 0],
            [27473, 32768, 0],
            [32487, 32768, 0],
            [7654, 32768, 0],
            [19473, 32768, 0],
            [29984, 32768, 0],
            [9961, 32768, 0],
            [30242, 32768, 0],
            [32117, 32768, 0],
        ],
        [
            [31548, 32768, 0],
            [1549, 32768, 0],
            [10130, 32768, 0],
            [16656, 32768, 0],
            [18591, 32768, 0],
            [26308, 32768, 0],
            [32537, 32768, 0],
            [5403, 32768, 0],
            [18096, 32768, 0],
            [30003, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [29957, 32768, 0],
            [5391, 32768, 0],
            [18039, 32768, 0],
            [23566, 32768, 0],
            [22431, 32768, 0],
            [25822, 32768, 0],
            [32197, 32768, 0],
            [3778, 32768, 0],
            [15336, 32768, 0],
            [28981, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [17920, 32768, 0],
            [1818, 32768, 0],
            [7282, 32768, 0],
            [25273, 32768, 0],
            [10923, 32768, 0],
            [31554, 32768, 0],
            [32624, 32768, 0],
            [1366, 32768, 0],
            [15628, 32768, 0],
            [30462, 32768, 0],
            [146, 32768, 0],
            [5132, 32768, 0],
            [31657, 32768, 0],
        ],
        [
            [6308, 32768, 0],
            [117, 32768, 0],
            [1638, 32768, 0],
            [2161, 32768, 0],
            [16384, 32768, 0],
            [10923, 32768, 0],
            [30247, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
    ],
    [
        [
            [30371, 32768, 0],
            [7570, 32768, 0],
            [13155, 32768, 0],
            [20751, 32768, 0],
            [20969, 32768, 0],
            [27067, 32768, 0],
            [32013, 32768, 0],
            [5495, 32768, 0],
            [17942, 32768, 0],
            [28280, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [31782, 32768, 0],
            [1836, 32768, 0],
            [10689, 32768, 0],
            [17604, 32768, 0],
            [21622, 32768, 0],
            [27518, 32768, 0],
            [32399, 32768, 0],
            [4419, 32768, 0],
            [16294, 32768, 0],
            [28345, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [31901, 32768, 0],
            [10311, 32768, 0],
            [18047, 32768, 0],
            [24806, 32768, 0],
            [23288, 32768, 0],
            [27914, 32768, 0],
            [32296, 32768, 0],
            [4215, 32768, 0],
            [15756, 32768, 0],
            [28341, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [26726, 32768, 0],
            [1045, 32768, 0],
            [11703, 32768, 0],
            [20590, 32768, 0],
            [18554, 32768, 0],
            [25970, 32768, 0],
            [31938, 32768, 0],
            [5583, 32768, 0],
            [21313, 32768, 0],
            [29390, 32768, 0],
            [641, 32768, 0],
            [22265, 32768, 0],
            [31452, 32768, 0],
        ],
        [
            [26584, 32768, 0],
            [188, 32768, 0],
            [8847, 32768, 0],
            [24519, 32768, 0],
            [22938, 32768, 0],
            [30583, 32768, 0],
            [32608, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
    ],
    [
        [
            [29614, 32768, 0],
            [9068, 32768, 0],
            [12924, 32768, 0],
            [19538, 32768, 0],
            [17737, 32768, 0],
            [24619, 32768, 0],
            [30642, 32768, 0],
            [4119, 32768, 0],
            [16026, 32768, 0],
            [25657, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [31957, 32768, 0],
            [3230, 32768, 0],
            [11153, 32768, 0],
            [18123, 32768, 0],
            [20143, 32768, 0],
            [26536, 32768, 0],
            [31986, 32768, 0],
            [3050, 32768, 0],
            [14603, 32768, 0],
            [25155, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [32363, 32768, 0],
            [10692, 32768, 0],
            [19090, 32768, 0],
            [24357, 32768, 0],
            [24442, 32768, 0],
            [28312, 32768, 0],
            [32169, 32768, 0],
            [3648, 32768, 0],
            [15690, 32768, 0],
            [26815, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [30669, 32768, 0],
            [3832, 32768, 0],
            [11663, 32768, 0],
            [18889, 32768, 0],
            [19782, 32768, 0],
            [23313, 32768, 0],
            [31330, 32768, 0],
            [5124, 32768, 0],
            [18719, 32768, 0],
            [28468, 32768, 0],
            [3082, 32768, 0],
            [20982, 32768, 0],
            [29443, 32768, 0],
        ],
        [
            [28573, 32768, 0],
            [3183, 32768, 0],
            [17802, 32768, 0],
            [25977, 32768, 0],
            [26677, 32768, 0],
            [27832, 32768, 0],
            [32387, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
    ],
    [
        [
            [26887, 32768, 0],
            [6729, 32768, 0],
            [10361, 32768, 0],
            [17442, 32768, 0],
            [15045, 32768, 0],
            [22478, 32768, 0],
            [29072, 32768, 0],
            [2713, 32768, 0],
            [11861, 32768, 0],
            [20773, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [31903, 32768, 0],
            [2044, 32768, 0],
            [7528, 32768, 0],
            [14618, 32768, 0],
            [16182, 32768, 0],
            [24168, 32768, 0],
            [31037, 32768, 0],
            [2786, 32768, 0],
            [11194, 32768, 0],
            [20155, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [32510, 32768, 0],
            [8430, 32768, 0],
            [17318, 32768, 0],
            [24154, 32768, 0],
            [23674, 32768, 0],
            [28789, 32768, 0],
            [32139, 32768, 0],
            [3440, 32768, 0],
            [13117, 32768, 0],
            [22702, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
        [
            [31671, 32768, 0],
            [2056, 32768, 0],
            [11746, 32768, 0],
            [16852, 32768, 0],
            [18635, 32768, 0],
            [24715, 32768, 0],
            [31484, 32768, 0],
            [4656, 32768, 0],
            [16074, 32768, 0],
            [24704, 32768, 0],
            [1806, 32768, 0],
            [14645, 32768, 0],
            [25336, 32768, 0],
        ],
        [
            [31539, 32768, 0],
            [8433, 32768, 0],
            [20576, 32768, 0],
            [27904, 32768, 0],
            [27852, 32768, 0],
            [30026, 32768, 0],
            [32441, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
            [16384, 32768, 0],
        ],
    ],
];

/// `Default_Eob_Pt_16_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 2 ][ 6 ]`
/// (§9.4). Codes `eob_pt_16` (the EOB position class for a 16-coefficient block).
pub const DEFAULT_EOB_PT_16_CDF: [[[[u16; 6]; 2]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [840, 1039, 1980, 4895, 32768, 0],
            [370, 671, 1883, 4471, 32768, 0],
        ],
        [
            [3247, 4950, 9688, 14563, 32768, 0],
            [1904, 3354, 7763, 14647, 32768, 0],
        ],
    ],
    [
        [
            [2125, 2551, 5165, 8946, 32768, 0],
            [513, 765, 1859, 6339, 32768, 0],
        ],
        [
            [7637, 9498, 14259, 19108, 32768, 0],
            [2497, 4096, 8866, 16993, 32768, 0],
        ],
    ],
    [
        [
            [4016, 4897, 8881, 14968, 32768, 0],
            [716, 1105, 2646, 10056, 32768, 0],
        ],
        [
            [11139, 13270, 18241, 23566, 32768, 0],
            [3192, 5032, 10297, 19755, 32768, 0],
        ],
    ],
    [
        [
            [6708, 8958, 14746, 22133, 32768, 0],
            [1222, 2074, 4783, 15410, 32768, 0],
        ],
        [
            [19575, 21766, 26044, 29709, 32768, 0],
            [7297, 10767, 19273, 28194, 32768, 0],
        ],
    ],
];

/// `Default_Eob_Pt_32_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 2 ][ 7 ]`
/// (§9.4). Codes `eob_pt_32`.
pub const DEFAULT_EOB_PT_32_CDF: [[[[u16; 7]; 2]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [400, 520, 977, 2102, 6542, 32768, 0],
            [210, 405, 1315, 3326, 7537, 32768, 0],
        ],
        [
            [2636, 4273, 7588, 11794, 20401, 32768, 0],
            [1786, 3179, 6902, 11357, 19054, 32768, 0],
        ],
    ],
    [
        [
            [989, 1249, 2019, 4151, 10785, 32768, 0],
            [313, 441, 1099, 2917, 8562, 32768, 0],
        ],
        [
            [8394, 10352, 13932, 18855, 26014, 32768, 0],
            [2578, 4124, 8181, 13670, 24234, 32768, 0],
        ],
    ],
    [
        [
            [2515, 3003, 4452, 8162, 16041, 32768, 0],
            [574, 821, 1836, 5089, 13128, 32768, 0],
        ],
        [
            [13468, 16303, 20361, 25105, 29281, 32768, 0],
            [3542, 5502, 10415, 16760, 25644, 32768, 0],
        ],
    ],
    [
        [
            [4617, 5709, 8446, 13584, 23135, 32768, 0],
            [1156, 1702, 3675, 9274, 20539, 32768, 0],
        ],
        [
            [22086, 24282, 27010, 29770, 31743, 32768, 0],
            [7699, 10897, 20891, 26926, 31628, 32768, 0],
        ],
    ],
];

/// `Default_Eob_Pt_64_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 2 ][ 8 ]`
/// (§9.4). Codes `eob_pt_64`.
pub const DEFAULT_EOB_PT_64_CDF: [[[[u16; 8]; 2]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [329, 498, 1101, 1784, 3265, 7758, 32768, 0],
            [335, 730, 1459, 5494, 8755, 12997, 32768, 0],
        ],
        [
            [3505, 5304, 10086, 13814, 17684, 23370, 32768, 0],
            [1563, 2700, 4876, 10911, 14706, 22480, 32768, 0],
        ],
    ],
    [
        [
            [1260, 1446, 2253, 3712, 6652, 13369, 32768, 0],
            [401, 605, 1029, 2563, 5845, 12626, 32768, 0],
        ],
        [
            [8609, 10612, 14624, 18714, 22614, 29024, 32768, 0],
            [1923, 3127, 5867, 9703, 14277, 27100, 32768, 0],
        ],
    ],
    [
        [
            [2374, 2772, 4583, 7276, 12288, 19706, 32768, 0],
            [497, 810, 1315, 3000, 7004, 15641, 32768, 0],
        ],
        [
            [15050, 17126, 21410, 24886, 28156, 30726, 32768, 0],
            [4034, 6290, 10235, 14982, 21214, 28491, 32768, 0],
        ],
    ],
    [
        [
            [6307, 7541, 12060, 16358, 22553, 27865, 32768, 0],
            [1289, 2320, 3971, 7926, 14153, 24291, 32768, 0],
        ],
        [
            [24212, 25708, 28268, 30035, 31307, 32049, 32768, 0],
            [8726, 12378, 19409, 26450, 30038, 32462, 32768, 0],
        ],
    ],
];

/// `Default_Eob_Pt_128_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 2 ][ 9 ]`
/// (§9.4). Codes `eob_pt_128`.
pub const DEFAULT_EOB_PT_128_CDF: [[[[u16; 9]; 2]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [219, 482, 1140, 2091, 3680, 6028, 12586, 32768, 0],
            [371, 699, 1254, 4830, 9479, 12562, 17497, 32768, 0],
        ],
        [
            [5245, 7456, 12880, 15852, 20033, 23932, 27608, 32768, 0],
            [2054, 3472, 5869, 14232, 18242, 20590, 26752, 32768, 0],
        ],
    ],
    [
        [
            [685, 933, 1488, 2714, 4766, 8562, 19254, 32768, 0],
            [217, 352, 618, 2303, 5261, 9969, 17472, 32768, 0],
        ],
        [
            [8045, 11200, 15497, 19595, 23948, 27408, 30938, 32768, 0],
            [2310, 4160, 7471, 14997, 17931, 20768, 30240, 32768, 0],
        ],
    ],
    [
        [
            [1366, 1738, 2527, 5016, 9355, 15797, 24643, 32768, 0],
            [354, 558, 944, 2760, 7287, 14037, 21779, 32768, 0],
        ],
        [
            [13627, 16246, 20173, 24429, 27948, 30415, 31863, 32768, 0],
            [6275, 9889, 14769, 23164, 27988, 30493, 32272, 32768, 0],
        ],
    ],
    [
        [
            [3472, 4885, 7489, 12481, 18517, 24536, 29635, 32768, 0],
            [886, 1731, 3271, 8469, 15569, 22126, 28383, 32768, 0],
        ],
        [
            [24313, 26062, 28385, 30107, 31217, 31898, 32345, 32768, 0],
            [9165, 13282, 21150, 30286, 31894, 32571, 32712, 32768, 0],
        ],
    ],
];

/// `Default_Eob_Pt_256_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 2 ][ 10 ]`
/// (§9.4). Codes `eob_pt_256`.
pub const DEFAULT_EOB_PT_256_CDF: [[[[u16; 10]; 2]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [310, 584, 1887, 3589, 6168, 8611, 11352, 15652, 32768, 0],
            [998, 1850, 2998, 5604, 17341, 19888, 22899, 25583, 32768, 0],
        ],
        [
            [2520, 3240, 5952, 8870, 12577, 17558, 19954, 24168, 32768, 0],
            [
                2203, 4130, 7435, 10739, 20652, 23681, 25609, 27261, 32768, 0,
            ],
        ],
    ],
    [
        [
            [1448, 2109, 4151, 6263, 9329, 13260, 17944, 23300, 32768, 0],
            [399, 1019, 1749, 3038, 10444, 15546, 22739, 27294, 32768, 0],
        ],
        [
            [
                6402, 8148, 12623, 15072, 18728, 22847, 26447, 29377, 32768, 0,
            ],
            [
                1674, 3252, 5734, 10159, 22397, 23802, 24821, 30940, 32768, 0,
            ],
        ],
    ],
    [
        [
            [3089, 3920, 6038, 9460, 14266, 19881, 25766, 29176, 32768, 0],
            [1084, 2358, 3488, 5122, 11483, 18103, 26023, 29799, 32768, 0],
        ],
        [
            [
                11514, 13794, 17480, 20754, 24361, 27378, 29492, 31277, 32768, 0,
            ],
            [
                6571, 9610, 15516, 21826, 29092, 30829, 31842, 32708, 32768, 0,
            ],
        ],
    ],
    [
        [
            [
                5348, 7113, 11820, 15924, 22106, 26777, 30334, 31757, 32768, 0,
            ],
            [2453, 4474, 6307, 8777, 16474, 22975, 29000, 31547, 32768, 0],
        ],
        [
            [
                23110, 24597, 27140, 28894, 30167, 30927, 31392, 32094, 32768, 0,
            ],
            [
                9998, 17661, 25178, 28097, 31308, 32038, 32403, 32695, 32768, 0,
            ],
        ],
    ],
];

/// `Default_Eob_Pt_512_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 11 ]`
/// (§9.4). Codes `eob_pt_512` (no per-`ptype`-pair axis).
pub const DEFAULT_EOB_PT_512_CDF: [[[u16; 11]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            641, 983, 3707, 5430, 10234, 14958, 18788, 23412, 26061, 32768, 0,
        ],
        [
            5095, 6446, 9996, 13354, 16017, 17986, 20919, 26129, 29140, 32768, 0,
        ],
    ],
    [
        [
            1230, 2278, 5035, 7776, 11871, 15346, 19590, 24584, 28749, 32768, 0,
        ],
        [
            7265, 9979, 15819, 19250, 21780, 23846, 26478, 28396, 31811, 32768, 0,
        ],
    ],
    [
        [
            2624, 3936, 6480, 9686, 13979, 17726, 23267, 28410, 31078, 32768, 0,
        ],
        [
            12015, 14769, 19588, 22052, 24222, 25812, 27300, 29219, 32114, 32768, 0,
        ],
    ],
    [
        [
            5927, 7809, 10923, 14597, 19439, 24135, 28456, 31142, 32060, 32768, 0,
        ],
        [
            21093, 23043, 25742, 27658, 29097, 29716, 30073, 30820, 31956, 32768, 0,
        ],
    ],
];

/// `Default_Eob_Pt_1024_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ 12 ]`
/// (§9.4). Codes `eob_pt_1024`.
pub const DEFAULT_EOB_PT_1024_CDF: [[[u16; 12]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            393, 421, 751, 1623, 3160, 6352, 13345, 18047, 22571, 25830, 32768, 0,
        ],
        [
            1865, 1988, 2930, 4242, 10533, 16538, 21354, 27255, 28546, 31784, 32768, 0,
        ],
    ],
    [
        [
            696, 948, 3145, 5702, 9706, 13217, 17851, 21856, 25692, 28034, 32768, 0,
        ],
        [
            2672, 3591, 9330, 17084, 22725, 24284, 26527, 28027, 28377, 30876, 32768, 0,
        ],
    ],
    [
        [
            2784, 3831, 7041, 10521, 14847, 18844, 23155, 26682, 29229, 31045, 32768, 0,
        ],
        [
            9577, 12466, 17739, 20750, 22061, 23215, 24601, 25483, 25843, 32056, 32768, 0,
        ],
    ],
    [
        [
            6698, 8334, 11961, 15762, 20186, 23862, 27434, 29326, 31082, 32050, 32768, 0,
        ],
        [
            20569, 22426, 25569, 26859, 28053, 28913, 29486, 29724, 29807, 32570, 32768, 0,
        ],
    ],
];

/// `Default_Eob_Extra_Cdf[ COEFF_CDF_Q_CTXS ][ TX_SIZES ][ PLANE_TYPES ]
/// [ EOB_COEF_CONTEXTS ][ 3 ]` (§9.4). Codes the binary `eob_extra` flag.
pub const DEFAULT_EOB_EXTRA_CDF: [[[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES];
    COEFF_CDF_Q_CTXS] = [
    [
        [
            [
                [16961, 32768, 0],
                [17223, 32768, 0],
                [7621, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [19069, 32768, 0],
                [22525, 32768, 0],
                [13377, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [20401, 32768, 0],
                [17025, 32768, 0],
                [12845, 32768, 0],
                [12873, 32768, 0],
                [14094, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [20681, 32768, 0],
                [20701, 32768, 0],
                [15250, 32768, 0],
                [15017, 32768, 0],
                [14928, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [23905, 32768, 0],
                [17194, 32768, 0],
                [16170, 32768, 0],
                [17695, 32768, 0],
                [13826, 32768, 0],
                [15810, 32768, 0],
                [12036, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [23959, 32768, 0],
                [20799, 32768, 0],
                [19021, 32768, 0],
                [16203, 32768, 0],
                [17886, 32768, 0],
                [14144, 32768, 0],
                [12010, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [27399, 32768, 0],
                [16327, 32768, 0],
                [18071, 32768, 0],
                [19584, 32768, 0],
                [20721, 32768, 0],
                [18432, 32768, 0],
                [19560, 32768, 0],
                [10150, 32768, 0],
                [8805, 32768, 0],
            ],
            [
                [24932, 32768, 0],
                [20833, 32768, 0],
                [12027, 32768, 0],
                [16670, 32768, 0],
                [19914, 32768, 0],
                [15106, 32768, 0],
                [17662, 32768, 0],
                [13783, 32768, 0],
                [28756, 32768, 0],
            ],
        ],
        [
            [
                [23406, 32768, 0],
                [21845, 32768, 0],
                [18432, 32768, 0],
                [16384, 32768, 0],
                [17096, 32768, 0],
                [12561, 32768, 0],
                [17320, 32768, 0],
                [22395, 32768, 0],
                [21370, 32768, 0],
            ],
            [
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [17471, 32768, 0],
                [20223, 32768, 0],
                [11357, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [20335, 32768, 0],
                [21667, 32768, 0],
                [14818, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [20430, 32768, 0],
                [20662, 32768, 0],
                [15367, 32768, 0],
                [16970, 32768, 0],
                [14657, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [22117, 32768, 0],
                [22028, 32768, 0],
                [18650, 32768, 0],
                [16042, 32768, 0],
                [15885, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [22409, 32768, 0],
                [21012, 32768, 0],
                [15650, 32768, 0],
                [17395, 32768, 0],
                [15469, 32768, 0],
                [20205, 32768, 0],
                [19511, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [24220, 32768, 0],
                [22480, 32768, 0],
                [17737, 32768, 0],
                [18916, 32768, 0],
                [19268, 32768, 0],
                [18412, 32768, 0],
                [18844, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [25991, 32768, 0],
                [20314, 32768, 0],
                [17731, 32768, 0],
                [19678, 32768, 0],
                [18649, 32768, 0],
                [17307, 32768, 0],
                [21798, 32768, 0],
                [17549, 32768, 0],
                [15630, 32768, 0],
            ],
            [
                [26585, 32768, 0],
                [21469, 32768, 0],
                [20432, 32768, 0],
                [17735, 32768, 0],
                [19280, 32768, 0],
                [15235, 32768, 0],
                [20297, 32768, 0],
                [22471, 32768, 0],
                [28997, 32768, 0],
            ],
        ],
        [
            [
                [26605, 32768, 0],
                [11304, 32768, 0],
                [16726, 32768, 0],
                [16560, 32768, 0],
                [20866, 32768, 0],
                [23524, 32768, 0],
                [19878, 32768, 0],
                [13469, 32768, 0],
                [23084, 32768, 0],
            ],
            [
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [18983, 32768, 0],
                [20512, 32768, 0],
                [14885, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [20090, 32768, 0],
                [19444, 32768, 0],
                [17286, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [19139, 32768, 0],
                [21487, 32768, 0],
                [18959, 32768, 0],
                [20910, 32768, 0],
                [19089, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [20536, 32768, 0],
                [20664, 32768, 0],
                [20625, 32768, 0],
                [19123, 32768, 0],
                [14862, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [19833, 32768, 0],
                [21502, 32768, 0],
                [17485, 32768, 0],
                [20267, 32768, 0],
                [18353, 32768, 0],
                [23329, 32768, 0],
                [21478, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [22041, 32768, 0],
                [23434, 32768, 0],
                [20001, 32768, 0],
                [20554, 32768, 0],
                [20951, 32768, 0],
                [20145, 32768, 0],
                [15562, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [23312, 32768, 0],
                [21607, 32768, 0],
                [16526, 32768, 0],
                [18957, 32768, 0],
                [18034, 32768, 0],
                [18934, 32768, 0],
                [24247, 32768, 0],
                [16921, 32768, 0],
                [17080, 32768, 0],
            ],
            [
                [26579, 32768, 0],
                [24910, 32768, 0],
                [18637, 32768, 0],
                [19800, 32768, 0],
                [20388, 32768, 0],
                [9887, 32768, 0],
                [15642, 32768, 0],
                [30198, 32768, 0],
                [24721, 32768, 0],
            ],
        ],
        [
            [
                [26998, 32768, 0],
                [16737, 32768, 0],
                [17838, 32768, 0],
                [18922, 32768, 0],
                [19515, 32768, 0],
                [18636, 32768, 0],
                [17333, 32768, 0],
                [15776, 32768, 0],
                [22658, 32768, 0],
            ],
            [
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [20177, 32768, 0],
                [20789, 32768, 0],
                [20262, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [21416, 32768, 0],
                [20855, 32768, 0],
                [23410, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [20238, 32768, 0],
                [21057, 32768, 0],
                [19159, 32768, 0],
                [22337, 32768, 0],
                [20159, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [20125, 32768, 0],
                [20559, 32768, 0],
                [21707, 32768, 0],
                [22296, 32768, 0],
                [17333, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [19941, 32768, 0],
                [20527, 32768, 0],
                [21470, 32768, 0],
                [22487, 32768, 0],
                [19558, 32768, 0],
                [22354, 32768, 0],
                [20331, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
            [
                [22752, 32768, 0],
                [25006, 32768, 0],
                [22075, 32768, 0],
                [21576, 32768, 0],
                [17740, 32768, 0],
                [21690, 32768, 0],
                [19211, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
        [
            [
                [21442, 32768, 0],
                [22358, 32768, 0],
                [18503, 32768, 0],
                [20291, 32768, 0],
                [19945, 32768, 0],
                [21294, 32768, 0],
                [21178, 32768, 0],
                [19400, 32768, 0],
                [10556, 32768, 0],
            ],
            [
                [24648, 32768, 0],
                [24949, 32768, 0],
                [20708, 32768, 0],
                [23905, 32768, 0],
                [20501, 32768, 0],
                [9558, 32768, 0],
                [9423, 32768, 0],
                [30365, 32768, 0],
                [19253, 32768, 0],
            ],
        ],
        [
            [
                [26064, 32768, 0],
                [22098, 32768, 0],
                [19613, 32768, 0],
                [20525, 32768, 0],
                [17595, 32768, 0],
                [16618, 32768, 0],
                [20497, 32768, 0],
                [18989, 32768, 0],
                [15513, 32768, 0],
            ],
            [
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
                [16384, 32768, 0],
            ],
        ],
    ],
];

/// `Default_Dc_Sign_Cdf[ COEFF_CDF_Q_CTXS ][ PLANE_TYPES ][ DC_SIGN_CONTEXTS ]
/// [ 3 ]` (§9.4). Codes the binary `dc_sign`. The §9.4 source gives the
/// frequencies as `128 * N` fixed-point products (preserved here); the four
/// `COEFF_CDF_Q_CTXS` slices are identical in the spec.
pub const DEFAULT_DC_SIGN_CDF: [[[[u16; 3]; DC_SIGN_CONTEXTS]; PLANE_TYPES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [128 * 125, 32768, 0],
            [128 * 102, 32768, 0],
            [128 * 147, 32768, 0],
        ],
        [
            [128 * 119, 32768, 0],
            [128 * 101, 32768, 0],
            [128 * 135, 32768, 0],
        ],
    ],
    [
        [
            [128 * 125, 32768, 0],
            [128 * 102, 32768, 0],
            [128 * 147, 32768, 0],
        ],
        [
            [128 * 119, 32768, 0],
            [128 * 101, 32768, 0],
            [128 * 135, 32768, 0],
        ],
    ],
    [
        [
            [128 * 125, 32768, 0],
            [128 * 102, 32768, 0],
            [128 * 147, 32768, 0],
        ],
        [
            [128 * 119, 32768, 0],
            [128 * 101, 32768, 0],
            [128 * 135, 32768, 0],
        ],
    ],
    [
        [
            [128 * 125, 32768, 0],
            [128 * 102, 32768, 0],
            [128 * 147, 32768, 0],
        ],
        [
            [128 * 119, 32768, 0],
            [128 * 101, 32768, 0],
            [128 * 135, 32768, 0],
        ],
    ],
];

// ---------------------------------------------------------------------
// Round 138 — coefficient `coeff_base_eob` sub-group default CDF (§9.4),
// the first of the `coeff_base` / `coeff_base_eob` / `coeff_br` braid
// reset by the §8.3.1 `init_coeff_cdfs` function. `coeff_base_eob`
// codes the base level of the last non-zero coefficient: the base level
// is `coeff_base_eob + 1`, and since this coefficient is known to be
// non-zero only base levels 1, 2, or 3 are coded — i.e. the cdf has
// three symbols (a 4-entry row: 3 cumulative frequencies + the §8.3
// adaptation counter). The remaining two tables of the braid
// (`Default_Coeff_Base_Cdf` and `Default_Coeff_Br_Cdf`) are deferred
// to later rounds. Transcribed verbatim from the §9.4 source; the
// outer `COEFF_CDF_Q_CTXS` axis is selected by `base_q_idx` at
// `init_coeff_cdfs`.
// ---------------------------------------------------------------------

/// `Default_Coeff_Base_Eob_Cdf[ COEFF_CDF_Q_CTXS ][ TX_SIZES ][ PLANE_TYPES ][ SIG_COEF_CONTEXTS_EOB ][ 4 ]`
/// (§9.4). Codes `coeff_base_eob`. Selected at `init_coeff_cdfs` by
/// `idx` (the [`coeff_cdf_q_ctx`] value), then by `txSzCtx`, `ptype`,
/// and the `coeff_base_eob` context (the latter is the
/// `get_coeff_base_ctx() - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`
/// reduction per §8.3.2).
pub const DEFAULT_COEFF_BASE_EOB_CDF: [[[[[u16; 4]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES];
    TX_SIZES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [
                [17837, 29055, 32768, 0],
                [29600, 31446, 32768, 0],
                [30844, 31878, 32768, 0],
                [24926, 28948, 32768, 0],
            ],
            [
                [21365, 30026, 32768, 0],
                [30512, 32423, 32768, 0],
                [31658, 32621, 32768, 0],
                [29630, 31881, 32768, 0],
            ],
        ],
        [
            [
                [5717, 26477, 32768, 0],
                [30491, 31703, 32768, 0],
                [31550, 32158, 32768, 0],
                [29648, 31491, 32768, 0],
            ],
            [
                [12608, 27820, 32768, 0],
                [30680, 32225, 32768, 0],
                [30809, 32335, 32768, 0],
                [31299, 32423, 32768, 0],
            ],
        ],
        [
            [
                [1786, 12612, 32768, 0],
                [30663, 31625, 32768, 0],
                [32339, 32468, 32768, 0],
                [31148, 31833, 32768, 0],
            ],
            [
                [18857, 23865, 32768, 0],
                [31428, 32428, 32768, 0],
                [31744, 32373, 32768, 0],
                [31775, 32526, 32768, 0],
            ],
        ],
        [
            [
                [1787, 2532, 32768, 0],
                [30832, 31662, 32768, 0],
                [31824, 32682, 32768, 0],
                [32133, 32569, 32768, 0],
            ],
            [
                [13751, 22235, 32768, 0],
                [32089, 32409, 32768, 0],
                [27084, 27920, 32768, 0],
                [29291, 32594, 32768, 0],
            ],
        ],
        [
            [
                [1725, 3449, 32768, 0],
                [31102, 31935, 32768, 0],
                [32457, 32613, 32768, 0],
                [32412, 32649, 32768, 0],
            ],
            [
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [17560, 29888, 32768, 0],
                [29671, 31549, 32768, 0],
                [31007, 32056, 32768, 0],
                [27286, 30006, 32768, 0],
            ],
            [
                [26594, 31212, 32768, 0],
                [31208, 32582, 32768, 0],
                [31835, 32637, 32768, 0],
                [30595, 32206, 32768, 0],
            ],
        ],
        [
            [
                [15239, 29932, 32768, 0],
                [31315, 32095, 32768, 0],
                [32130, 32434, 32768, 0],
                [30864, 31996, 32768, 0],
            ],
            [
                [26279, 30968, 32768, 0],
                [31142, 32495, 32768, 0],
                [31713, 32540, 32768, 0],
                [31929, 32594, 32768, 0],
            ],
        ],
        [
            [
                [2644, 25198, 32768, 0],
                [32038, 32451, 32768, 0],
                [32639, 32695, 32768, 0],
                [32166, 32518, 32768, 0],
            ],
            [
                [17187, 27668, 32768, 0],
                [31714, 32550, 32768, 0],
                [32283, 32678, 32768, 0],
                [31930, 32563, 32768, 0],
            ],
        ],
        [
            [
                [1044, 2257, 32768, 0],
                [30755, 31923, 32768, 0],
                [32208, 32693, 32768, 0],
                [32244, 32615, 32768, 0],
            ],
            [
                [21317, 26207, 32768, 0],
                [29133, 30868, 32768, 0],
                [29311, 31231, 32768, 0],
                [29657, 31087, 32768, 0],
            ],
        ],
        [
            [
                [478, 1834, 32768, 0],
                [31005, 31987, 32768, 0],
                [32317, 32724, 32768, 0],
                [30865, 32648, 32768, 0],
            ],
            [
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [20092, 30774, 32768, 0],
                [30695, 32020, 32768, 0],
                [31131, 32103, 32768, 0],
                [28666, 30870, 32768, 0],
            ],
            [
                [27258, 31095, 32768, 0],
                [31804, 32623, 32768, 0],
                [31763, 32528, 32768, 0],
                [31438, 32506, 32768, 0],
            ],
        ],
        [
            [
                [18049, 30489, 32768, 0],
                [31706, 32286, 32768, 0],
                [32163, 32473, 32768, 0],
                [31550, 32184, 32768, 0],
            ],
            [
                [27116, 30842, 32768, 0],
                [31971, 32598, 32768, 0],
                [32088, 32576, 32768, 0],
                [32067, 32664, 32768, 0],
            ],
        ],
        [
            [
                [12854, 29093, 32768, 0],
                [32272, 32558, 32768, 0],
                [32667, 32729, 32768, 0],
                [32306, 32585, 32768, 0],
            ],
            [
                [25476, 30366, 32768, 0],
                [32169, 32687, 32768, 0],
                [32479, 32689, 32768, 0],
                [31673, 32634, 32768, 0],
            ],
        ],
        [
            [
                [2809, 19301, 32768, 0],
                [32205, 32622, 32768, 0],
                [32338, 32730, 32768, 0],
                [31786, 32616, 32768, 0],
            ],
            [
                [22737, 29105, 32768, 0],
                [30810, 32362, 32768, 0],
                [30014, 32627, 32768, 0],
                [30528, 32574, 32768, 0],
            ],
        ],
        [
            [
                [935, 3382, 32768, 0],
                [30789, 31909, 32768, 0],
                [32466, 32756, 32768, 0],
                [30860, 32513, 32768, 0],
            ],
            [
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [22497, 31198, 32768, 0],
                [31715, 32495, 32768, 0],
                [31606, 32337, 32768, 0],
                [30388, 31990, 32768, 0],
            ],
            [
                [27877, 31584, 32768, 0],
                [32170, 32728, 32768, 0],
                [32155, 32688, 32768, 0],
                [32219, 32702, 32768, 0],
            ],
        ],
        [
            [
                [21457, 31043, 32768, 0],
                [31951, 32483, 32768, 0],
                [32153, 32562, 32768, 0],
                [31473, 32215, 32768, 0],
            ],
            [
                [27558, 31151, 32768, 0],
                [32020, 32640, 32768, 0],
                [32097, 32575, 32768, 0],
                [32242, 32719, 32768, 0],
            ],
        ],
        [
            [
                [19980, 30591, 32768, 0],
                [32219, 32597, 32768, 0],
                [32581, 32706, 32768, 0],
                [31803, 32287, 32768, 0],
            ],
            [
                [26473, 30507, 32768, 0],
                [32431, 32723, 32768, 0],
                [32196, 32611, 32768, 0],
                [31588, 32528, 32768, 0],
            ],
        ],
        [
            [
                [24647, 30463, 32768, 0],
                [32412, 32695, 32768, 0],
                [32468, 32720, 32768, 0],
                [31269, 32523, 32768, 0],
            ],
            [
                [28482, 31505, 32768, 0],
                [32152, 32701, 32768, 0],
                [31732, 32598, 32768, 0],
                [31767, 32712, 32768, 0],
            ],
        ],
        [
            [
                [12358, 24977, 32768, 0],
                [31331, 32385, 32768, 0],
                [32634, 32756, 32768, 0],
                [30411, 32548, 32768, 0],
            ],
            [
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
                [10923, 21845, 32768, 0],
            ],
        ],
    ],
];

// ---------------------------------------------------------------------
// Round 139 — coefficient `coeff_base` sub-group default CDF (§9.4),
// the second of the `coeff_base` / `coeff_base_eob` / `coeff_br` braid
// reset by the §8.3.1 `init_coeff_cdfs` function. `coeff_base` codes
// the base level of each non-EOB coefficient: the four-symbol alphabet
// is `0..3`, so the cdf is a 5-entry row (4 cumulative frequencies +
// the §8.3 adaptation counter). The last remaining table of the braid
// (`Default_Coeff_Br_Cdf`) is deferred to a later round. Transcribed
// verbatim from the §9.4 source; the outer `COEFF_CDF_Q_CTXS` axis is
// selected by `base_q_idx` at `init_coeff_cdfs`.
// ---------------------------------------------------------------------

/// `Default_Coeff_Base_Cdf[ COEFF_CDF_Q_CTXS ][ TX_SIZES ][ PLANE_TYPES ][ SIG_COEF_CONTEXTS ][ 5 ]`
/// (§9.4). Codes `coeff_base`. Selected at `init_coeff_cdfs` by
/// `idx` (the [`coeff_cdf_q_ctx`] value), then by `txSzCtx`, `ptype`,
/// and the `coeff_base` context (the §8.3.2 `get_coeff_base_ctx()`
/// result, in `0..SIG_COEF_CONTEXTS`).
///
/// Declared `static` rather than `const` because the table is large
/// (1680 5-entry rows = 16800 bytes) and `clippy::large_const_arrays`
/// flags const arrays of this size as a per-use copy hazard. The
/// shape and read semantics are otherwise identical to the other
/// `Default_*_Cdf` arrays in this module.
pub static DEFAULT_COEFF_BASE_CDF: [[[[[u16; 5]; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES];
    COEFF_CDF_Q_CTXS] = [
    [
        [
            [
                [4034, 8930, 12727, 32768, 0],
                [18082, 29741, 31877, 32768, 0],
                [12596, 26124, 30493, 32768, 0],
                [9446, 21118, 27005, 32768, 0],
                [6308, 15141, 21279, 32768, 0],
                [2463, 6357, 9783, 32768, 0],
                [20667, 30546, 31929, 32768, 0],
                [13043, 26123, 30134, 32768, 0],
                [8151, 18757, 24778, 32768, 0],
                [5255, 12839, 18632, 32768, 0],
                [2820, 7206, 11161, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [15736, 27553, 30604, 32768, 0],
                [11210, 23794, 28787, 32768, 0],
                [5947, 13874, 19701, 32768, 0],
                [4215, 9323, 13891, 32768, 0],
                [2833, 6462, 10059, 32768, 0],
                [19605, 30393, 31582, 32768, 0],
                [13523, 26252, 30248, 32768, 0],
                [8446, 18622, 24512, 32768, 0],
                [3818, 10343, 15974, 32768, 0],
                [1481, 4117, 6796, 32768, 0],
                [22649, 31302, 32190, 32768, 0],
                [14829, 27127, 30449, 32768, 0],
                [8313, 17702, 23304, 32768, 0],
                [3022, 8301, 12786, 32768, 0],
                [1536, 4412, 7184, 32768, 0],
                [22354, 29774, 31372, 32768, 0],
                [14723, 25472, 29214, 32768, 0],
                [6673, 13745, 18662, 32768, 0],
                [2068, 5766, 9322, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [6302, 16444, 21761, 32768, 0],
                [23040, 31538, 32475, 32768, 0],
                [15196, 28452, 31496, 32768, 0],
                [10020, 22946, 28514, 32768, 0],
                [6533, 16862, 23501, 32768, 0],
                [3538, 9816, 15076, 32768, 0],
                [24444, 31875, 32525, 32768, 0],
                [15881, 28924, 31635, 32768, 0],
                [9922, 22873, 28466, 32768, 0],
                [6527, 16966, 23691, 32768, 0],
                [4114, 11303, 17220, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [20201, 30770, 32209, 32768, 0],
                [14754, 28071, 31258, 32768, 0],
                [8378, 20186, 26517, 32768, 0],
                [5916, 15299, 21978, 32768, 0],
                [4268, 11583, 17901, 32768, 0],
                [24361, 32025, 32581, 32768, 0],
                [18673, 30105, 31943, 32768, 0],
                [10196, 22244, 27576, 32768, 0],
                [5495, 14349, 20417, 32768, 0],
                [2676, 7415, 11498, 32768, 0],
                [24678, 31958, 32585, 32768, 0],
                [18629, 29906, 31831, 32768, 0],
                [9364, 20724, 26315, 32768, 0],
                [4641, 12318, 18094, 32768, 0],
                [2758, 7387, 11579, 32768, 0],
                [25433, 31842, 32469, 32768, 0],
                [18795, 29289, 31411, 32768, 0],
                [7644, 17584, 23592, 32768, 0],
                [3408, 9014, 15047, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [4536, 10072, 14001, 32768, 0],
                [25459, 31416, 32206, 32768, 0],
                [16605, 28048, 30818, 32768, 0],
                [11008, 22857, 27719, 32768, 0],
                [6915, 16268, 22315, 32768, 0],
                [2625, 6812, 10537, 32768, 0],
                [24257, 31788, 32499, 32768, 0],
                [16880, 29454, 31879, 32768, 0],
                [11958, 25054, 29778, 32768, 0],
                [7916, 18718, 25084, 32768, 0],
                [3383, 8777, 13446, 32768, 0],
                [22720, 31603, 32393, 32768, 0],
                [14960, 28125, 31335, 32768, 0],
                [9731, 22210, 27928, 32768, 0],
                [6304, 15832, 22277, 32768, 0],
                [2910, 7818, 12166, 32768, 0],
                [20375, 30627, 32131, 32768, 0],
                [13904, 27284, 30887, 32768, 0],
                [9368, 21558, 27144, 32768, 0],
                [5937, 14966, 21119, 32768, 0],
                [2667, 7225, 11319, 32768, 0],
                [23970, 31470, 32378, 32768, 0],
                [17173, 29734, 32018, 32768, 0],
                [12795, 25441, 29965, 32768, 0],
                [8981, 19680, 25893, 32768, 0],
                [4728, 11372, 16902, 32768, 0],
                [24287, 31797, 32439, 32768, 0],
                [16703, 29145, 31696, 32768, 0],
                [10833, 23554, 28725, 32768, 0],
                [6468, 16566, 23057, 32768, 0],
                [2415, 6562, 10278, 32768, 0],
                [26610, 32395, 32659, 32768, 0],
                [18590, 30498, 32117, 32768, 0],
                [12420, 25756, 29950, 32768, 0],
                [7639, 18746, 24710, 32768, 0],
                [3001, 8086, 12347, 32768, 0],
                [25076, 32064, 32580, 32768, 0],
                [17946, 30128, 32028, 32768, 0],
                [12024, 24985, 29378, 32768, 0],
                [7517, 18390, 24304, 32768, 0],
                [3243, 8781, 13331, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [6037, 16771, 21957, 32768, 0],
                [24774, 31704, 32426, 32768, 0],
                [16830, 28589, 31056, 32768, 0],
                [10602, 22828, 27760, 32768, 0],
                [6733, 16829, 23071, 32768, 0],
                [3250, 8914, 13556, 32768, 0],
                [25582, 32220, 32668, 32768, 0],
                [18659, 30342, 32223, 32768, 0],
                [12546, 26149, 30515, 32768, 0],
                [8420, 20451, 26801, 32768, 0],
                [4636, 12420, 18344, 32768, 0],
                [27581, 32362, 32639, 32768, 0],
                [18987, 30083, 31978, 32768, 0],
                [11327, 24248, 29084, 32768, 0],
                [7264, 17719, 24120, 32768, 0],
                [3995, 10768, 16169, 32768, 0],
                [25893, 31831, 32487, 32768, 0],
                [16577, 28587, 31379, 32768, 0],
                [10189, 22748, 28182, 32768, 0],
                [6832, 17094, 23556, 32768, 0],
                [3708, 10110, 15334, 32768, 0],
                [25904, 32282, 32656, 32768, 0],
                [19721, 30792, 32276, 32768, 0],
                [12819, 26243, 30411, 32768, 0],
                [8572, 20614, 26891, 32768, 0],
                [5364, 14059, 20467, 32768, 0],
                [26580, 32438, 32677, 32768, 0],
                [20852, 31225, 32340, 32768, 0],
                [12435, 25700, 29967, 32768, 0],
                [8691, 20825, 26976, 32768, 0],
                [4446, 12209, 17269, 32768, 0],
                [27350, 32429, 32696, 32768, 0],
                [21372, 30977, 32272, 32768, 0],
                [12673, 25270, 29853, 32768, 0],
                [9208, 20925, 26640, 32768, 0],
                [5018, 13351, 18732, 32768, 0],
                [27351, 32479, 32713, 32768, 0],
                [21398, 31209, 32387, 32768, 0],
                [12162, 25047, 29842, 32768, 0],
                [7896, 18691, 25319, 32768, 0],
                [4670, 12882, 18881, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [5487, 10460, 13708, 32768, 0],
                [21597, 28303, 30674, 32768, 0],
                [11037, 21953, 26476, 32768, 0],
                [8147, 17962, 22952, 32768, 0],
                [5242, 13061, 18532, 32768, 0],
                [1889, 5208, 8182, 32768, 0],
                [26774, 32133, 32590, 32768, 0],
                [17844, 29564, 31767, 32768, 0],
                [11690, 24438, 29171, 32768, 0],
                [7542, 18215, 24459, 32768, 0],
                [2993, 8050, 12319, 32768, 0],
                [28023, 32328, 32591, 32768, 0],
                [18651, 30126, 31954, 32768, 0],
                [12164, 25146, 29589, 32768, 0],
                [7762, 18530, 24771, 32768, 0],
                [3492, 9183, 13920, 32768, 0],
                [27591, 32008, 32491, 32768, 0],
                [17149, 28853, 31510, 32768, 0],
                [11485, 24003, 28860, 32768, 0],
                [7697, 18086, 24210, 32768, 0],
                [3075, 7999, 12218, 32768, 0],
                [28268, 32482, 32654, 32768, 0],
                [19631, 31051, 32404, 32768, 0],
                [13860, 27260, 31020, 32768, 0],
                [9605, 21613, 27594, 32768, 0],
                [4876, 12162, 17908, 32768, 0],
                [27248, 32316, 32576, 32768, 0],
                [18955, 30457, 32075, 32768, 0],
                [11824, 23997, 28795, 32768, 0],
                [7346, 18196, 24647, 32768, 0],
                [3403, 9247, 14111, 32768, 0],
                [29711, 32655, 32735, 32768, 0],
                [21169, 31394, 32417, 32768, 0],
                [13487, 27198, 30957, 32768, 0],
                [8828, 21683, 27614, 32768, 0],
                [4270, 11451, 17038, 32768, 0],
                [28708, 32578, 32731, 32768, 0],
                [20120, 31241, 32482, 32768, 0],
                [13692, 27550, 31321, 32768, 0],
                [9418, 22514, 28439, 32768, 0],
                [4999, 13283, 19462, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [5673, 14302, 19711, 32768, 0],
                [26251, 30701, 31834, 32768, 0],
                [12782, 23783, 27803, 32768, 0],
                [9127, 20657, 25808, 32768, 0],
                [6368, 16208, 21462, 32768, 0],
                [2465, 7177, 10822, 32768, 0],
                [29961, 32563, 32719, 32768, 0],
                [18318, 29891, 31949, 32768, 0],
                [11361, 24514, 29357, 32768, 0],
                [7900, 19603, 25607, 32768, 0],
                [4002, 10590, 15546, 32768, 0],
                [29637, 32310, 32595, 32768, 0],
                [18296, 29913, 31809, 32768, 0],
                [10144, 21515, 26871, 32768, 0],
                [5358, 14322, 20394, 32768, 0],
                [3067, 8362, 13346, 32768, 0],
                [28652, 32470, 32676, 32768, 0],
                [17538, 30771, 32209, 32768, 0],
                [13924, 26882, 30494, 32768, 0],
                [10496, 22837, 27869, 32768, 0],
                [7236, 16396, 21621, 32768, 0],
                [30743, 32687, 32746, 32768, 0],
                [23006, 31676, 32489, 32768, 0],
                [14494, 27828, 31120, 32768, 0],
                [10174, 22801, 28352, 32768, 0],
                [6242, 15281, 21043, 32768, 0],
                [25817, 32243, 32720, 32768, 0],
                [18618, 31367, 32325, 32768, 0],
                [13997, 28318, 31878, 32768, 0],
                [12255, 26534, 31383, 32768, 0],
                [9561, 21588, 28450, 32768, 0],
                [28188, 32635, 32724, 32768, 0],
                [22060, 32365, 32728, 32768, 0],
                [18102, 30690, 32528, 32768, 0],
                [14196, 28864, 31999, 32768, 0],
                [12262, 25792, 30865, 32768, 0],
                [24176, 32109, 32628, 32768, 0],
                [18280, 29681, 31963, 32768, 0],
                [10205, 23703, 29664, 32768, 0],
                [7889, 20025, 27676, 32768, 0],
                [6060, 16743, 23970, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [5141, 7096, 8260, 32768, 0],
                [27186, 29022, 29789, 32768, 0],
                [6668, 12568, 15682, 32768, 0],
                [2172, 6181, 8638, 32768, 0],
                [1126, 3379, 4531, 32768, 0],
                [443, 1361, 2254, 32768, 0],
                [26083, 31153, 32436, 32768, 0],
                [13486, 24603, 28483, 32768, 0],
                [6508, 14840, 19910, 32768, 0],
                [3386, 8800, 13286, 32768, 0],
                [1530, 4322, 7054, 32768, 0],
                [29639, 32080, 32548, 32768, 0],
                [15897, 27552, 30290, 32768, 0],
                [8588, 20047, 25383, 32768, 0],
                [4889, 13339, 19269, 32768, 0],
                [2240, 6871, 10498, 32768, 0],
                [28165, 32197, 32517, 32768, 0],
                [20735, 30427, 31568, 32768, 0],
                [14325, 24671, 27692, 32768, 0],
                [5119, 12554, 17805, 32768, 0],
                [1810, 5441, 8261, 32768, 0],
                [31212, 32724, 32748, 32768, 0],
                [23352, 31766, 32545, 32768, 0],
                [14669, 27570, 31059, 32768, 0],
                [8492, 20894, 27272, 32768, 0],
                [3644, 10194, 15204, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [2461, 7013, 9371, 32768, 0],
                [24749, 29600, 30986, 32768, 0],
                [9466, 19037, 22417, 32768, 0],
                [3584, 9280, 14400, 32768, 0],
                [1505, 3929, 5433, 32768, 0],
                [677, 1500, 2736, 32768, 0],
                [23987, 30702, 32117, 32768, 0],
                [13554, 24571, 29263, 32768, 0],
                [6211, 14556, 21155, 32768, 0],
                [3135, 10972, 15625, 32768, 0],
                [2435, 7127, 11427, 32768, 0],
                [31300, 32532, 32550, 32768, 0],
                [14757, 30365, 31954, 32768, 0],
                [4405, 11612, 18553, 32768, 0],
                [580, 4132, 7322, 32768, 0],
                [1695, 10169, 14124, 32768, 0],
                [30008, 32282, 32591, 32768, 0],
                [19244, 30108, 31748, 32768, 0],
                [11180, 24158, 29555, 32768, 0],
                [5650, 14972, 19209, 32768, 0],
                [2114, 5109, 8456, 32768, 0],
                [31856, 32716, 32748, 32768, 0],
                [23012, 31664, 32572, 32768, 0],
                [13694, 26656, 30636, 32768, 0],
                [8142, 19508, 26093, 32768, 0],
                [4253, 10955, 16724, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [601, 983, 1311, 32768, 0],
                [18725, 23406, 28087, 32768, 0],
                [5461, 8192, 10923, 32768, 0],
                [3781, 15124, 21425, 32768, 0],
                [2587, 7761, 12072, 32768, 0],
                [106, 458, 810, 32768, 0],
                [22282, 29710, 31894, 32768, 0],
                [8508, 20926, 25984, 32768, 0],
                [3726, 12713, 18083, 32768, 0],
                [1620, 7112, 10893, 32768, 0],
                [729, 2236, 3495, 32768, 0],
                [30163, 32474, 32684, 32768, 0],
                [18304, 30464, 32000, 32768, 0],
                [11443, 26526, 29647, 32768, 0],
                [6007, 15292, 21299, 32768, 0],
                [2234, 6703, 8937, 32768, 0],
                [30954, 32177, 32571, 32768, 0],
                [17363, 29562, 31076, 32768, 0],
                [9686, 22464, 27410, 32768, 0],
                [8192, 16384, 21390, 32768, 0],
                [1755, 8046, 11264, 32768, 0],
                [31168, 32734, 32748, 32768, 0],
                [22486, 31441, 32471, 32768, 0],
                [12833, 25627, 29738, 32768, 0],
                [6980, 17379, 23122, 32768, 0],
                [3111, 8887, 13479, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [6041, 11854, 15927, 32768, 0],
                [20326, 30905, 32251, 32768, 0],
                [14164, 26831, 30725, 32768, 0],
                [9760, 20647, 26585, 32768, 0],
                [6416, 14953, 21219, 32768, 0],
                [2966, 7151, 10891, 32768, 0],
                [23567, 31374, 32254, 32768, 0],
                [14978, 27416, 30946, 32768, 0],
                [9434, 20225, 26254, 32768, 0],
                [6658, 14558, 20535, 32768, 0],
                [3916, 8677, 12989, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [18088, 29545, 31587, 32768, 0],
                [13062, 25843, 30073, 32768, 0],
                [8940, 16827, 22251, 32768, 0],
                [7654, 13220, 17973, 32768, 0],
                [5733, 10316, 14456, 32768, 0],
                [22879, 31388, 32114, 32768, 0],
                [15215, 27993, 30955, 32768, 0],
                [9397, 19445, 24978, 32768, 0],
                [3442, 9813, 15344, 32768, 0],
                [1368, 3936, 6532, 32768, 0],
                [25494, 32033, 32406, 32768, 0],
                [16772, 27963, 30718, 32768, 0],
                [9419, 18165, 23260, 32768, 0],
                [2677, 7501, 11797, 32768, 0],
                [1516, 4344, 7170, 32768, 0],
                [26556, 31454, 32101, 32768, 0],
                [17128, 27035, 30108, 32768, 0],
                [8324, 15344, 20249, 32768, 0],
                [1903, 5696, 9469, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8455, 19003, 24368, 32768, 0],
                [23563, 32021, 32604, 32768, 0],
                [16237, 29446, 31935, 32768, 0],
                [10724, 23999, 29358, 32768, 0],
                [6725, 17528, 24416, 32768, 0],
                [3927, 10927, 16825, 32768, 0],
                [26313, 32288, 32634, 32768, 0],
                [17430, 30095, 32095, 32768, 0],
                [11116, 24606, 29679, 32768, 0],
                [7195, 18384, 25269, 32768, 0],
                [4726, 12852, 19315, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [22822, 31648, 32483, 32768, 0],
                [16724, 29633, 31929, 32768, 0],
                [10261, 23033, 28725, 32768, 0],
                [7029, 17840, 24528, 32768, 0],
                [4867, 13886, 21502, 32768, 0],
                [25298, 31892, 32491, 32768, 0],
                [17809, 29330, 31512, 32768, 0],
                [9668, 21329, 26579, 32768, 0],
                [4774, 12956, 18976, 32768, 0],
                [2322, 7030, 11540, 32768, 0],
                [25472, 31920, 32543, 32768, 0],
                [17957, 29387, 31632, 32768, 0],
                [9196, 20593, 26400, 32768, 0],
                [4680, 12705, 19202, 32768, 0],
                [2917, 8456, 13436, 32768, 0],
                [26471, 32059, 32574, 32768, 0],
                [18458, 29783, 31909, 32768, 0],
                [8400, 19464, 25956, 32768, 0],
                [3812, 10973, 17206, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [6779, 13743, 17678, 32768, 0],
                [24806, 31797, 32457, 32768, 0],
                [17616, 29047, 31372, 32768, 0],
                [11063, 23175, 28003, 32768, 0],
                [6521, 16110, 22324, 32768, 0],
                [2764, 7504, 11654, 32768, 0],
                [25266, 32367, 32637, 32768, 0],
                [19054, 30553, 32175, 32768, 0],
                [12139, 25212, 29807, 32768, 0],
                [7311, 18162, 24704, 32768, 0],
                [3397, 9164, 14074, 32768, 0],
                [25988, 32208, 32522, 32768, 0],
                [16253, 28912, 31526, 32768, 0],
                [9151, 21387, 27372, 32768, 0],
                [5688, 14915, 21496, 32768, 0],
                [2717, 7627, 12004, 32768, 0],
                [23144, 31855, 32443, 32768, 0],
                [16070, 28491, 31325, 32768, 0],
                [8702, 20467, 26517, 32768, 0],
                [5243, 13956, 20367, 32768, 0],
                [2621, 7335, 11567, 32768, 0],
                [26636, 32340, 32630, 32768, 0],
                [19990, 31050, 32341, 32768, 0],
                [13243, 26105, 30315, 32768, 0],
                [8588, 19521, 25918, 32768, 0],
                [4717, 11585, 17304, 32768, 0],
                [25844, 32292, 32582, 32768, 0],
                [19090, 30635, 32097, 32768, 0],
                [11963, 24546, 28939, 32768, 0],
                [6218, 16087, 22354, 32768, 0],
                [2340, 6608, 10426, 32768, 0],
                [28046, 32576, 32694, 32768, 0],
                [21178, 31313, 32296, 32768, 0],
                [13486, 26184, 29870, 32768, 0],
                [7149, 17871, 23723, 32768, 0],
                [2833, 7958, 12259, 32768, 0],
                [27710, 32528, 32686, 32768, 0],
                [20674, 31076, 32268, 32768, 0],
                [12413, 24955, 29243, 32768, 0],
                [6676, 16927, 23097, 32768, 0],
                [2966, 8333, 12919, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8639, 19339, 24429, 32768, 0],
                [24404, 31837, 32525, 32768, 0],
                [16997, 29425, 31784, 32768, 0],
                [11253, 24234, 29149, 32768, 0],
                [6751, 17394, 24028, 32768, 0],
                [3490, 9830, 15191, 32768, 0],
                [26283, 32471, 32714, 32768, 0],
                [19599, 31168, 32442, 32768, 0],
                [13146, 26954, 30893, 32768, 0],
                [8214, 20588, 26890, 32768, 0],
                [4699, 13081, 19300, 32768, 0],
                [28212, 32458, 32669, 32768, 0],
                [18594, 30316, 32100, 32768, 0],
                [11219, 24408, 29234, 32768, 0],
                [6865, 17656, 24149, 32768, 0],
                [3678, 10362, 16006, 32768, 0],
                [25825, 32136, 32616, 32768, 0],
                [17313, 29853, 32021, 32768, 0],
                [11197, 24471, 29472, 32768, 0],
                [6947, 17781, 24405, 32768, 0],
                [3768, 10660, 16261, 32768, 0],
                [27352, 32500, 32706, 32768, 0],
                [20850, 31468, 32469, 32768, 0],
                [14021, 27707, 31133, 32768, 0],
                [8964, 21748, 27838, 32768, 0],
                [5437, 14665, 21187, 32768, 0],
                [26304, 32492, 32698, 32768, 0],
                [20409, 31380, 32385, 32768, 0],
                [13682, 27222, 30632, 32768, 0],
                [8974, 21236, 26685, 32768, 0],
                [4234, 11665, 16934, 32768, 0],
                [26273, 32357, 32711, 32768, 0],
                [20672, 31242, 32441, 32768, 0],
                [14172, 27254, 30902, 32768, 0],
                [9870, 21898, 27275, 32768, 0],
                [5164, 13506, 19270, 32768, 0],
                [26725, 32459, 32728, 32768, 0],
                [20991, 31442, 32527, 32768, 0],
                [13071, 26434, 30811, 32768, 0],
                [8184, 20090, 26742, 32768, 0],
                [4803, 13255, 19895, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [7555, 14942, 18501, 32768, 0],
                [24410, 31178, 32287, 32768, 0],
                [14394, 26738, 30253, 32768, 0],
                [8413, 19554, 25195, 32768, 0],
                [4766, 12924, 18785, 32768, 0],
                [2029, 5806, 9207, 32768, 0],
                [26776, 32364, 32663, 32768, 0],
                [18732, 29967, 31931, 32768, 0],
                [11005, 23786, 28852, 32768, 0],
                [6466, 16909, 23510, 32768, 0],
                [3044, 8638, 13419, 32768, 0],
                [29208, 32582, 32704, 32768, 0],
                [20068, 30857, 32208, 32768, 0],
                [12003, 25085, 29595, 32768, 0],
                [6947, 17750, 24189, 32768, 0],
                [3245, 9103, 14007, 32768, 0],
                [27359, 32465, 32669, 32768, 0],
                [19421, 30614, 32174, 32768, 0],
                [11915, 25010, 29579, 32768, 0],
                [6950, 17676, 24074, 32768, 0],
                [3007, 8473, 13096, 32768, 0],
                [29002, 32676, 32735, 32768, 0],
                [22102, 31849, 32576, 32768, 0],
                [14408, 28009, 31405, 32768, 0],
                [9027, 21679, 27931, 32768, 0],
                [4694, 12678, 18748, 32768, 0],
                [28216, 32528, 32682, 32768, 0],
                [20849, 31264, 32318, 32768, 0],
                [12756, 25815, 29751, 32768, 0],
                [7565, 18801, 24923, 32768, 0],
                [3509, 9533, 14477, 32768, 0],
                [30133, 32687, 32739, 32768, 0],
                [23063, 31910, 32515, 32768, 0],
                [14588, 28051, 31132, 32768, 0],
                [9085, 21649, 27457, 32768, 0],
                [4261, 11654, 17264, 32768, 0],
                [29518, 32691, 32748, 32768, 0],
                [22451, 31959, 32613, 32768, 0],
                [14864, 28722, 31700, 32768, 0],
                [9695, 22964, 28716, 32768, 0],
                [4932, 13358, 19502, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [6465, 16958, 21688, 32768, 0],
                [25199, 31514, 32360, 32768, 0],
                [14774, 27149, 30607, 32768, 0],
                [9257, 21438, 26972, 32768, 0],
                [5723, 15183, 21882, 32768, 0],
                [3150, 8879, 13731, 32768, 0],
                [26989, 32262, 32682, 32768, 0],
                [17396, 29937, 32085, 32768, 0],
                [11387, 24901, 29784, 32768, 0],
                [7289, 18821, 25548, 32768, 0],
                [3734, 10577, 16086, 32768, 0],
                [29728, 32501, 32695, 32768, 0],
                [17431, 29701, 31903, 32768, 0],
                [9921, 22826, 28300, 32768, 0],
                [5896, 15434, 22068, 32768, 0],
                [3430, 9646, 14757, 32768, 0],
                [28614, 32511, 32705, 32768, 0],
                [19364, 30638, 32263, 32768, 0],
                [13129, 26254, 30402, 32768, 0],
                [8754, 20484, 26440, 32768, 0],
                [4378, 11607, 17110, 32768, 0],
                [30292, 32671, 32744, 32768, 0],
                [21780, 31603, 32501, 32768, 0],
                [14314, 27829, 31291, 32768, 0],
                [9611, 22327, 28263, 32768, 0],
                [4890, 13087, 19065, 32768, 0],
                [25862, 32567, 32733, 32768, 0],
                [20794, 32050, 32567, 32768, 0],
                [17243, 30625, 32254, 32768, 0],
                [13283, 27628, 31474, 32768, 0],
                [9669, 22532, 28918, 32768, 0],
                [27435, 32697, 32748, 32768, 0],
                [24922, 32390, 32714, 32768, 0],
                [21449, 31504, 32536, 32768, 0],
                [16392, 29729, 31832, 32768, 0],
                [11692, 24884, 29076, 32768, 0],
                [24193, 32290, 32735, 32768, 0],
                [18909, 31104, 32563, 32768, 0],
                [12236, 26841, 31403, 32768, 0],
                [8171, 21840, 29082, 32768, 0],
                [7224, 17280, 25275, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [3078, 6839, 9890, 32768, 0],
                [13837, 20450, 24479, 32768, 0],
                [5914, 14222, 19328, 32768, 0],
                [3866, 10267, 14762, 32768, 0],
                [2612, 7208, 11042, 32768, 0],
                [1067, 2991, 4776, 32768, 0],
                [25817, 31646, 32529, 32768, 0],
                [13708, 26338, 30385, 32768, 0],
                [7328, 18585, 24870, 32768, 0],
                [4691, 13080, 19276, 32768, 0],
                [1825, 5253, 8352, 32768, 0],
                [29386, 32315, 32624, 32768, 0],
                [17160, 29001, 31360, 32768, 0],
                [9602, 21862, 27396, 32768, 0],
                [5915, 15772, 22148, 32768, 0],
                [2786, 7779, 12047, 32768, 0],
                [29246, 32450, 32663, 32768, 0],
                [18696, 29929, 31818, 32768, 0],
                [10510, 23369, 28560, 32768, 0],
                [6229, 16499, 23125, 32768, 0],
                [2608, 7448, 11705, 32768, 0],
                [30753, 32710, 32748, 32768, 0],
                [21638, 31487, 32503, 32768, 0],
                [12937, 26854, 30870, 32768, 0],
                [8182, 20596, 26970, 32768, 0],
                [3637, 10269, 15497, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [5244, 12150, 16906, 32768, 0],
                [20486, 26858, 29701, 32768, 0],
                [7756, 18317, 23735, 32768, 0],
                [3452, 9256, 13146, 32768, 0],
                [2020, 5206, 8229, 32768, 0],
                [1801, 4993, 7903, 32768, 0],
                [27051, 31858, 32531, 32768, 0],
                [15988, 27531, 30619, 32768, 0],
                [9188, 21484, 26719, 32768, 0],
                [6273, 17186, 23800, 32768, 0],
                [3108, 9355, 14764, 32768, 0],
                [31076, 32520, 32680, 32768, 0],
                [18119, 30037, 31850, 32768, 0],
                [10244, 22969, 27472, 32768, 0],
                [4692, 14077, 19273, 32768, 0],
                [3694, 11677, 17556, 32768, 0],
                [30060, 32581, 32720, 32768, 0],
                [21011, 30775, 32120, 32768, 0],
                [11931, 24820, 29289, 32768, 0],
                [7119, 17662, 24356, 32768, 0],
                [3833, 10706, 16304, 32768, 0],
                [31954, 32731, 32748, 32768, 0],
                [23913, 31724, 32489, 32768, 0],
                [15520, 28060, 31286, 32768, 0],
                [11517, 23008, 28571, 32768, 0],
                [6193, 14508, 20629, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [1035, 2807, 4156, 32768, 0],
                [13162, 18138, 20939, 32768, 0],
                [2696, 6633, 8755, 32768, 0],
                [1373, 4161, 6853, 32768, 0],
                [1099, 2746, 4716, 32768, 0],
                [340, 1021, 1599, 32768, 0],
                [22826, 30419, 32135, 32768, 0],
                [10395, 21762, 26942, 32768, 0],
                [4726, 12407, 17361, 32768, 0],
                [2447, 7080, 10593, 32768, 0],
                [1227, 3717, 6011, 32768, 0],
                [28156, 31424, 31934, 32768, 0],
                [16915, 27754, 30373, 32768, 0],
                [9148, 20990, 26431, 32768, 0],
                [5950, 15515, 21148, 32768, 0],
                [2492, 7327, 11526, 32768, 0],
                [30602, 32477, 32670, 32768, 0],
                [20026, 29955, 31568, 32768, 0],
                [11220, 23628, 28105, 32768, 0],
                [6652, 17019, 22973, 32768, 0],
                [3064, 8536, 13043, 32768, 0],
                [31769, 32724, 32748, 32768, 0],
                [22230, 30887, 32373, 32768, 0],
                [12234, 25079, 29731, 32768, 0],
                [7326, 18816, 25353, 32768, 0],
                [3933, 10907, 16616, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [8896, 16227, 20630, 32768, 0],
                [23629, 31782, 32527, 32768, 0],
                [15173, 27755, 31321, 32768, 0],
                [10158, 21233, 27382, 32768, 0],
                [6420, 14857, 21558, 32768, 0],
                [3269, 8155, 12646, 32768, 0],
                [24835, 32009, 32496, 32768, 0],
                [16509, 28421, 31579, 32768, 0],
                [10957, 21514, 27418, 32768, 0],
                [7881, 15930, 22096, 32768, 0],
                [5388, 10960, 15918, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [20745, 30773, 32093, 32768, 0],
                [15200, 27221, 30861, 32768, 0],
                [13032, 20873, 25667, 32768, 0],
                [12285, 18663, 23494, 32768, 0],
                [11563, 17481, 21489, 32768, 0],
                [26260, 31982, 32320, 32768, 0],
                [15397, 28083, 31100, 32768, 0],
                [9742, 19217, 24824, 32768, 0],
                [3261, 9629, 15362, 32768, 0],
                [1480, 4322, 7499, 32768, 0],
                [27599, 32256, 32460, 32768, 0],
                [16857, 27659, 30774, 32768, 0],
                [9551, 18290, 23748, 32768, 0],
                [3052, 8933, 14103, 32768, 0],
                [2021, 5910, 9787, 32768, 0],
                [29005, 32015, 32392, 32768, 0],
                [17677, 27694, 30863, 32768, 0],
                [9204, 17356, 23219, 32768, 0],
                [2403, 7516, 12814, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [10808, 22056, 26896, 32768, 0],
                [25739, 32313, 32676, 32768, 0],
                [17288, 30203, 32221, 32768, 0],
                [11359, 24878, 29896, 32768, 0],
                [6949, 17767, 24893, 32768, 0],
                [4287, 11796, 18071, 32768, 0],
                [27880, 32521, 32705, 32768, 0],
                [19038, 31004, 32414, 32768, 0],
                [12564, 26345, 30768, 32768, 0],
                [8269, 19947, 26779, 32768, 0],
                [5674, 14657, 21674, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [25742, 32319, 32671, 32768, 0],
                [19557, 31164, 32454, 32768, 0],
                [13381, 26381, 30755, 32768, 0],
                [10101, 21466, 26722, 32768, 0],
                [9209, 19650, 26825, 32768, 0],
                [27107, 31917, 32432, 32768, 0],
                [18056, 28893, 31203, 32768, 0],
                [10200, 21434, 26764, 32768, 0],
                [4660, 12913, 19502, 32768, 0],
                [2368, 6930, 12504, 32768, 0],
                [26960, 32158, 32613, 32768, 0],
                [18628, 30005, 32031, 32768, 0],
                [10233, 22442, 28232, 32768, 0],
                [5471, 14630, 21516, 32768, 0],
                [3235, 10767, 17109, 32768, 0],
                [27696, 32440, 32692, 32768, 0],
                [20032, 31167, 32438, 32768, 0],
                [8700, 21341, 28442, 32768, 0],
                [5662, 14831, 21795, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [9704, 17294, 21132, 32768, 0],
                [26762, 32278, 32633, 32768, 0],
                [18382, 29620, 31819, 32768, 0],
                [10891, 23475, 28723, 32768, 0],
                [6358, 16583, 23309, 32768, 0],
                [3248, 9118, 14141, 32768, 0],
                [27204, 32573, 32699, 32768, 0],
                [19818, 30824, 32329, 32768, 0],
                [11772, 25120, 30041, 32768, 0],
                [6995, 18033, 25039, 32768, 0],
                [3752, 10442, 16098, 32768, 0],
                [27222, 32256, 32559, 32768, 0],
                [15356, 28399, 31475, 32768, 0],
                [8821, 20635, 27057, 32768, 0],
                [5511, 14404, 21239, 32768, 0],
                [2935, 8222, 13051, 32768, 0],
                [24875, 32120, 32529, 32768, 0],
                [15233, 28265, 31445, 32768, 0],
                [8605, 20570, 26932, 32768, 0],
                [5431, 14413, 21196, 32768, 0],
                [2994, 8341, 13223, 32768, 0],
                [28201, 32604, 32700, 32768, 0],
                [21041, 31446, 32456, 32768, 0],
                [13221, 26213, 30475, 32768, 0],
                [8255, 19385, 26037, 32768, 0],
                [4930, 12585, 18830, 32768, 0],
                [28768, 32448, 32627, 32768, 0],
                [19705, 30561, 32021, 32768, 0],
                [11572, 23589, 28220, 32768, 0],
                [5532, 15034, 21446, 32768, 0],
                [2460, 7150, 11456, 32768, 0],
                [29874, 32619, 32699, 32768, 0],
                [21621, 31071, 32201, 32768, 0],
                [12511, 24747, 28992, 32768, 0],
                [6281, 16395, 22748, 32768, 0],
                [3246, 9278, 14497, 32768, 0],
                [29715, 32625, 32712, 32768, 0],
                [20958, 31011, 32283, 32768, 0],
                [11233, 23671, 28806, 32768, 0],
                [6012, 16128, 22868, 32768, 0],
                [3427, 9851, 15414, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [11016, 22111, 26794, 32768, 0],
                [25946, 32357, 32677, 32768, 0],
                [17890, 30452, 32252, 32768, 0],
                [11678, 25142, 29816, 32768, 0],
                [6720, 17534, 24584, 32768, 0],
                [4230, 11665, 17820, 32768, 0],
                [28400, 32623, 32747, 32768, 0],
                [21164, 31668, 32575, 32768, 0],
                [13572, 27388, 31182, 32768, 0],
                [8234, 20750, 27358, 32768, 0],
                [5065, 14055, 20897, 32768, 0],
                [28981, 32547, 32705, 32768, 0],
                [18681, 30543, 32239, 32768, 0],
                [10919, 24075, 29286, 32768, 0],
                [6431, 17199, 24077, 32768, 0],
                [3819, 10464, 16618, 32768, 0],
                [26870, 32467, 32693, 32768, 0],
                [19041, 30831, 32347, 32768, 0],
                [11794, 25211, 30016, 32768, 0],
                [6888, 18019, 24970, 32768, 0],
                [4370, 12363, 18992, 32768, 0],
                [29578, 32670, 32744, 32768, 0],
                [23159, 32007, 32613, 32768, 0],
                [15315, 28669, 31676, 32768, 0],
                [9298, 22607, 28782, 32768, 0],
                [6144, 15913, 22968, 32768, 0],
                [28110, 32499, 32669, 32768, 0],
                [21574, 30937, 32015, 32768, 0],
                [12759, 24818, 28727, 32768, 0],
                [6545, 16761, 23042, 32768, 0],
                [3649, 10597, 16833, 32768, 0],
                [28163, 32552, 32728, 32768, 0],
                [22101, 31469, 32464, 32768, 0],
                [13160, 25472, 30143, 32768, 0],
                [7303, 18684, 25468, 32768, 0],
                [5241, 13975, 20955, 32768, 0],
                [28400, 32631, 32744, 32768, 0],
                [22104, 31793, 32603, 32768, 0],
                [13557, 26571, 30846, 32768, 0],
                [7749, 19861, 26675, 32768, 0],
                [4873, 14030, 21234, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [9800, 17635, 21073, 32768, 0],
                [26153, 31885, 32527, 32768, 0],
                [15038, 27852, 31006, 32768, 0],
                [8718, 20564, 26486, 32768, 0],
                [5128, 14076, 20514, 32768, 0],
                [2636, 7566, 11925, 32768, 0],
                [27551, 32504, 32701, 32768, 0],
                [18310, 30054, 32100, 32768, 0],
                [10211, 23420, 29082, 32768, 0],
                [6222, 16876, 23916, 32768, 0],
                [3462, 9954, 15498, 32768, 0],
                [29991, 32633, 32721, 32768, 0],
                [19883, 30751, 32201, 32768, 0],
                [11141, 24184, 29285, 32768, 0],
                [6420, 16940, 23774, 32768, 0],
                [3392, 9753, 15118, 32768, 0],
                [28465, 32616, 32712, 32768, 0],
                [19850, 30702, 32244, 32768, 0],
                [10983, 24024, 29223, 32768, 0],
                [6294, 16770, 23582, 32768, 0],
                [3244, 9283, 14509, 32768, 0],
                [30023, 32717, 32748, 32768, 0],
                [22940, 32032, 32626, 32768, 0],
                [14282, 27928, 31473, 32768, 0],
                [8562, 21327, 27914, 32768, 0],
                [4846, 13393, 19919, 32768, 0],
                [29981, 32590, 32695, 32768, 0],
                [20465, 30963, 32166, 32768, 0],
                [11479, 23579, 28195, 32768, 0],
                [5916, 15648, 22073, 32768, 0],
                [3031, 8605, 13398, 32768, 0],
                [31146, 32691, 32739, 32768, 0],
                [23106, 31724, 32444, 32768, 0],
                [13783, 26738, 30439, 32768, 0],
                [7852, 19468, 25807, 32768, 0],
                [3860, 11124, 16853, 32768, 0],
                [31014, 32724, 32748, 32768, 0],
                [23629, 32109, 32628, 32768, 0],
                [14747, 28115, 31403, 32768, 0],
                [8545, 21242, 27478, 32768, 0],
                [4574, 12781, 19067, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [9185, 19694, 24688, 32768, 0],
                [26081, 31985, 32621, 32768, 0],
                [16015, 29000, 31787, 32768, 0],
                [10542, 23690, 29206, 32768, 0],
                [6732, 17945, 24677, 32768, 0],
                [3916, 11039, 16722, 32768, 0],
                [28224, 32566, 32744, 32768, 0],
                [19100, 31138, 32485, 32768, 0],
                [12528, 26620, 30879, 32768, 0],
                [7741, 20277, 26885, 32768, 0],
                [4566, 12845, 18990, 32768, 0],
                [29933, 32593, 32718, 32768, 0],
                [17670, 30333, 32155, 32768, 0],
                [10385, 23600, 28909, 32768, 0],
                [6243, 16236, 22407, 32768, 0],
                [3976, 10389, 16017, 32768, 0],
                [28377, 32561, 32738, 32768, 0],
                [19366, 31175, 32482, 32768, 0],
                [13327, 27175, 31094, 32768, 0],
                [8258, 20769, 27143, 32768, 0],
                [4703, 13198, 19527, 32768, 0],
                [31086, 32706, 32748, 32768, 0],
                [22853, 31902, 32583, 32768, 0],
                [14759, 28186, 31419, 32768, 0],
                [9284, 22382, 28348, 32768, 0],
                [5585, 15192, 21868, 32768, 0],
                [28291, 32652, 32746, 32768, 0],
                [19849, 32107, 32571, 32768, 0],
                [14834, 26818, 29214, 32768, 0],
                [10306, 22594, 28672, 32768, 0],
                [6615, 17384, 23384, 32768, 0],
                [28947, 32604, 32745, 32768, 0],
                [25625, 32289, 32646, 32768, 0],
                [18758, 28672, 31403, 32768, 0],
                [10017, 23430, 28523, 32768, 0],
                [6862, 15269, 22131, 32768, 0],
                [23933, 32509, 32739, 32768, 0],
                [19927, 31495, 32631, 32768, 0],
                [11903, 26023, 30621, 32768, 0],
                [7026, 20094, 27252, 32768, 0],
                [5998, 18106, 24437, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [4456, 11274, 15533, 32768, 0],
                [21219, 29079, 31616, 32768, 0],
                [11173, 23774, 28567, 32768, 0],
                [7282, 18293, 24263, 32768, 0],
                [4890, 13286, 19115, 32768, 0],
                [1890, 5508, 8659, 32768, 0],
                [26651, 32136, 32647, 32768, 0],
                [14630, 28254, 31455, 32768, 0],
                [8716, 21287, 27395, 32768, 0],
                [5615, 15331, 22008, 32768, 0],
                [2675, 7700, 12150, 32768, 0],
                [29954, 32526, 32690, 32768, 0],
                [16126, 28982, 31633, 32768, 0],
                [9030, 21361, 27352, 32768, 0],
                [5411, 14793, 21271, 32768, 0],
                [2943, 8422, 13163, 32768, 0],
                [29539, 32601, 32730, 32768, 0],
                [18125, 30385, 32201, 32768, 0],
                [10422, 24090, 29468, 32768, 0],
                [6468, 17487, 24438, 32768, 0],
                [2970, 8653, 13531, 32768, 0],
                [30912, 32715, 32748, 32768, 0],
                [20666, 31373, 32497, 32768, 0],
                [12509, 26640, 30917, 32768, 0],
                [8058, 20629, 27290, 32768, 0],
                [4231, 12006, 18052, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [10202, 20633, 25484, 32768, 0],
                [27336, 31445, 32352, 32768, 0],
                [12420, 24384, 28552, 32768, 0],
                [7648, 18115, 23856, 32768, 0],
                [5662, 14341, 19902, 32768, 0],
                [3611, 10328, 15390, 32768, 0],
                [30945, 32616, 32736, 32768, 0],
                [18682, 30505, 32253, 32768, 0],
                [11513, 25336, 30203, 32768, 0],
                [7449, 19452, 26148, 32768, 0],
                [4482, 13051, 18886, 32768, 0],
                [32022, 32690, 32747, 32768, 0],
                [18578, 30501, 32146, 32768, 0],
                [11249, 23368, 28631, 32768, 0],
                [5645, 16958, 22158, 32768, 0],
                [5009, 11444, 16637, 32768, 0],
                [31357, 32710, 32748, 32768, 0],
                [21552, 31494, 32504, 32768, 0],
                [13891, 27677, 31340, 32768, 0],
                [9051, 22098, 28172, 32768, 0],
                [5190, 13377, 19486, 32768, 0],
                [32364, 32740, 32748, 32768, 0],
                [24839, 31907, 32551, 32768, 0],
                [17160, 28779, 31696, 32768, 0],
                [12452, 24137, 29602, 32768, 0],
                [6165, 15389, 22477, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [2575, 7281, 11077, 32768, 0],
                [14002, 20866, 25402, 32768, 0],
                [6343, 15056, 19658, 32768, 0],
                [4474, 11858, 17041, 32768, 0],
                [2865, 8299, 12534, 32768, 0],
                [1344, 3949, 6391, 32768, 0],
                [24720, 31239, 32459, 32768, 0],
                [12585, 25356, 29968, 32768, 0],
                [7181, 18246, 24444, 32768, 0],
                [5025, 13667, 19885, 32768, 0],
                [2521, 7304, 11605, 32768, 0],
                [29908, 32252, 32584, 32768, 0],
                [17421, 29156, 31575, 32768, 0],
                [9889, 22188, 27782, 32768, 0],
                [5878, 15647, 22123, 32768, 0],
                [2814, 8665, 13323, 32768, 0],
                [30183, 32568, 32713, 32768, 0],
                [18528, 30195, 32049, 32768, 0],
                [10982, 24606, 29657, 32768, 0],
                [6957, 18165, 25231, 32768, 0],
                [3508, 10118, 15468, 32768, 0],
                [31761, 32736, 32748, 32768, 0],
                [21041, 31328, 32546, 32768, 0],
                [12568, 26732, 31166, 32768, 0],
                [8052, 20720, 27733, 32768, 0],
                [4336, 12192, 18396, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [7062, 16472, 22319, 32768, 0],
                [24538, 32261, 32674, 32768, 0],
                [13675, 28041, 31779, 32768, 0],
                [8590, 20674, 27631, 32768, 0],
                [5685, 14675, 22013, 32768, 0],
                [3655, 9898, 15731, 32768, 0],
                [26493, 32418, 32658, 32768, 0],
                [16376, 29342, 32090, 32768, 0],
                [10594, 22649, 28970, 32768, 0],
                [8176, 17170, 24303, 32768, 0],
                [5605, 12694, 19139, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [23888, 31902, 32542, 32768, 0],
                [18612, 29687, 31987, 32768, 0],
                [16245, 24852, 29249, 32768, 0],
                [15765, 22608, 27559, 32768, 0],
                [19895, 24699, 27510, 32768, 0],
                [28401, 32212, 32457, 32768, 0],
                [15274, 27825, 30980, 32768, 0],
                [9364, 18128, 24332, 32768, 0],
                [2283, 8193, 15082, 32768, 0],
                [1228, 3972, 7881, 32768, 0],
                [29455, 32469, 32620, 32768, 0],
                [17981, 28245, 31388, 32768, 0],
                [10921, 20098, 26240, 32768, 0],
                [3743, 11829, 18657, 32768, 0],
                [2374, 9593, 15715, 32768, 0],
                [31068, 32466, 32635, 32768, 0],
                [20321, 29572, 31971, 32768, 0],
                [10771, 20255, 27119, 32768, 0],
                [2795, 10410, 17361, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [9320, 22102, 27840, 32768, 0],
                [27057, 32464, 32724, 32768, 0],
                [16331, 30268, 32309, 32768, 0],
                [10319, 23935, 29720, 32768, 0],
                [6189, 16448, 24106, 32768, 0],
                [3589, 10884, 18808, 32768, 0],
                [29026, 32624, 32748, 32768, 0],
                [19226, 31507, 32587, 32768, 0],
                [12692, 26921, 31203, 32768, 0],
                [7049, 19532, 27635, 32768, 0],
                [7727, 15669, 23252, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [28056, 32625, 32748, 32768, 0],
                [22383, 32075, 32669, 32768, 0],
                [15417, 27098, 31749, 32768, 0],
                [18127, 26493, 27190, 32768, 0],
                [5461, 16384, 21845, 32768, 0],
                [27982, 32091, 32584, 32768, 0],
                [19045, 29868, 31972, 32768, 0],
                [10397, 22266, 27932, 32768, 0],
                [5990, 13697, 21500, 32768, 0],
                [1792, 6912, 15104, 32768, 0],
                [28198, 32501, 32718, 32768, 0],
                [21534, 31521, 32569, 32768, 0],
                [11109, 25217, 30017, 32768, 0],
                [5671, 15124, 26151, 32768, 0],
                [4681, 14043, 18725, 32768, 0],
                [28688, 32580, 32741, 32768, 0],
                [22576, 32079, 32661, 32768, 0],
                [10627, 22141, 28340, 32768, 0],
                [9362, 14043, 28087, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [7754, 16948, 22142, 32768, 0],
                [25670, 32330, 32691, 32768, 0],
                [15663, 29225, 31994, 32768, 0],
                [9878, 23288, 29158, 32768, 0],
                [6419, 17088, 24336, 32768, 0],
                [3859, 11003, 17039, 32768, 0],
                [27562, 32595, 32725, 32768, 0],
                [17575, 30588, 32399, 32768, 0],
                [10819, 24838, 30309, 32768, 0],
                [7124, 18686, 25916, 32768, 0],
                [4479, 12688, 19340, 32768, 0],
                [28385, 32476, 32673, 32768, 0],
                [15306, 29005, 31938, 32768, 0],
                [8937, 21615, 28322, 32768, 0],
                [5982, 15603, 22786, 32768, 0],
                [3620, 10267, 16136, 32768, 0],
                [27280, 32464, 32667, 32768, 0],
                [15607, 29160, 32004, 32768, 0],
                [9091, 22135, 28740, 32768, 0],
                [6232, 16632, 24020, 32768, 0],
                [4047, 11377, 17672, 32768, 0],
                [29220, 32630, 32718, 32768, 0],
                [19650, 31220, 32462, 32768, 0],
                [13050, 26312, 30827, 32768, 0],
                [9228, 20870, 27468, 32768, 0],
                [6146, 15149, 21971, 32768, 0],
                [30169, 32481, 32623, 32768, 0],
                [17212, 29311, 31554, 32768, 0],
                [9911, 21311, 26882, 32768, 0],
                [4487, 13314, 20372, 32768, 0],
                [2570, 7772, 12889, 32768, 0],
                [30924, 32613, 32708, 32768, 0],
                [19490, 30206, 32107, 32768, 0],
                [11232, 23998, 29276, 32768, 0],
                [6769, 17955, 25035, 32768, 0],
                [4398, 12623, 19214, 32768, 0],
                [30609, 32627, 32722, 32768, 0],
                [19370, 30582, 32287, 32768, 0],
                [10457, 23619, 29409, 32768, 0],
                [6443, 17637, 24834, 32768, 0],
                [4645, 13236, 20106, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8626, 20271, 26216, 32768, 0],
                [26707, 32406, 32711, 32768, 0],
                [16999, 30329, 32286, 32768, 0],
                [11445, 25123, 30286, 32768, 0],
                [6411, 18828, 25601, 32768, 0],
                [6801, 12458, 20248, 32768, 0],
                [29918, 32682, 32748, 32768, 0],
                [20649, 31739, 32618, 32768, 0],
                [12879, 27773, 31581, 32768, 0],
                [7896, 21751, 28244, 32768, 0],
                [5260, 14870, 23698, 32768, 0],
                [29252, 32593, 32731, 32768, 0],
                [17072, 30460, 32294, 32768, 0],
                [10653, 24143, 29365, 32768, 0],
                [6536, 17490, 23983, 32768, 0],
                [4929, 13170, 20085, 32768, 0],
                [28137, 32518, 32715, 32768, 0],
                [18171, 30784, 32407, 32768, 0],
                [11437, 25436, 30459, 32768, 0],
                [7252, 18534, 26176, 32768, 0],
                [4126, 13353, 20978, 32768, 0],
                [31162, 32726, 32748, 32768, 0],
                [23017, 32222, 32701, 32768, 0],
                [15629, 29233, 32046, 32768, 0],
                [9387, 22621, 29480, 32768, 0],
                [6922, 17616, 25010, 32768, 0],
                [28838, 32265, 32614, 32768, 0],
                [19701, 30206, 31920, 32768, 0],
                [11214, 22410, 27933, 32768, 0],
                [5320, 14177, 23034, 32768, 0],
                [5049, 12881, 17827, 32768, 0],
                [27484, 32471, 32734, 32768, 0],
                [21076, 31526, 32561, 32768, 0],
                [12707, 26303, 31211, 32768, 0],
                [8169, 21722, 28219, 32768, 0],
                [6045, 19406, 27042, 32768, 0],
                [27753, 32572, 32745, 32768, 0],
                [20832, 31878, 32653, 32768, 0],
                [13250, 27356, 31674, 32768, 0],
                [7718, 21508, 29858, 32768, 0],
                [7209, 18350, 25559, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [7876, 16901, 21741, 32768, 0],
                [24001, 31898, 32625, 32768, 0],
                [14529, 27959, 31451, 32768, 0],
                [8273, 20818, 27258, 32768, 0],
                [5278, 14673, 21510, 32768, 0],
                [2983, 8843, 14039, 32768, 0],
                [28016, 32574, 32732, 32768, 0],
                [17471, 30306, 32301, 32768, 0],
                [10224, 24063, 29728, 32768, 0],
                [6602, 17954, 25052, 32768, 0],
                [4002, 11585, 17759, 32768, 0],
                [30190, 32634, 32739, 32768, 0],
                [17497, 30282, 32270, 32768, 0],
                [10229, 23729, 29538, 32768, 0],
                [6344, 17211, 24440, 32768, 0],
                [3849, 11189, 17108, 32768, 0],
                [28570, 32583, 32726, 32768, 0],
                [17521, 30161, 32238, 32768, 0],
                [10153, 23565, 29378, 32768, 0],
                [6455, 17341, 24443, 32768, 0],
                [3907, 11042, 17024, 32768, 0],
                [30689, 32715, 32748, 32768, 0],
                [21546, 31840, 32610, 32768, 0],
                [13547, 27581, 31459, 32768, 0],
                [8912, 21757, 28309, 32768, 0],
                [5548, 15080, 22046, 32768, 0],
                [30783, 32540, 32685, 32768, 0],
                [17540, 29528, 31668, 32768, 0],
                [10160, 21468, 26783, 32768, 0],
                [4724, 13393, 20054, 32768, 0],
                [2702, 8174, 13102, 32768, 0],
                [31648, 32686, 32742, 32768, 0],
                [20954, 31094, 32337, 32768, 0],
                [12420, 25698, 30179, 32768, 0],
                [7304, 19320, 26248, 32768, 0],
                [4366, 12261, 18864, 32768, 0],
                [31581, 32723, 32748, 32768, 0],
                [21373, 31586, 32525, 32768, 0],
                [12744, 26625, 30885, 32768, 0],
                [7431, 20322, 26950, 32768, 0],
                [4692, 13323, 20111, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [7833, 18369, 24095, 32768, 0],
                [26650, 32273, 32702, 32768, 0],
                [16371, 29961, 32191, 32768, 0],
                [11055, 24082, 29629, 32768, 0],
                [6892, 18644, 25400, 32768, 0],
                [5006, 13057, 19240, 32768, 0],
                [29834, 32666, 32748, 32768, 0],
                [19577, 31335, 32570, 32768, 0],
                [12253, 26509, 31122, 32768, 0],
                [7991, 20772, 27711, 32768, 0],
                [5677, 15910, 23059, 32768, 0],
                [30109, 32532, 32720, 32768, 0],
                [16747, 30166, 32252, 32768, 0],
                [10134, 23542, 29184, 32768, 0],
                [5791, 16176, 23556, 32768, 0],
                [4362, 10414, 17284, 32768, 0],
                [29492, 32626, 32748, 32768, 0],
                [19894, 31402, 32525, 32768, 0],
                [12942, 27071, 30869, 32768, 0],
                [8346, 21216, 27405, 32768, 0],
                [6572, 17087, 23859, 32768, 0],
                [32035, 32735, 32748, 32768, 0],
                [22957, 31838, 32618, 32768, 0],
                [14724, 28572, 31772, 32768, 0],
                [10364, 23999, 29553, 32768, 0],
                [7004, 18433, 25655, 32768, 0],
                [27528, 32277, 32681, 32768, 0],
                [16959, 31171, 32096, 32768, 0],
                [10486, 23593, 27962, 32768, 0],
                [8192, 16384, 23211, 32768, 0],
                [8937, 17873, 20852, 32768, 0],
                [27715, 32002, 32615, 32768, 0],
                [15073, 29491, 31676, 32768, 0],
                [11264, 24576, 28672, 32768, 0],
                [2341, 18725, 23406, 32768, 0],
                [7282, 18204, 25486, 32768, 0],
                [28547, 32213, 32657, 32768, 0],
                [20788, 29773, 32239, 32768, 0],
                [6780, 21469, 30508, 32768, 0],
                [5958, 14895, 23831, 32768, 0],
                [16384, 21845, 27307, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [5992, 14304, 19765, 32768, 0],
                [22612, 31238, 32456, 32768, 0],
                [13456, 27162, 31087, 32768, 0],
                [8001, 20062, 26504, 32768, 0],
                [5168, 14105, 20764, 32768, 0],
                [2632, 7771, 12385, 32768, 0],
                [27034, 32344, 32709, 32768, 0],
                [15850, 29415, 31997, 32768, 0],
                [9494, 22776, 28841, 32768, 0],
                [6151, 16830, 23969, 32768, 0],
                [3461, 10039, 15722, 32768, 0],
                [30134, 32569, 32731, 32768, 0],
                [15638, 29422, 31945, 32768, 0],
                [9150, 21865, 28218, 32768, 0],
                [5647, 15719, 22676, 32768, 0],
                [3402, 9772, 15477, 32768, 0],
                [28530, 32586, 32735, 32768, 0],
                [17139, 30298, 32292, 32768, 0],
                [10200, 24039, 29685, 32768, 0],
                [6419, 17674, 24786, 32768, 0],
                [3544, 10225, 15824, 32768, 0],
                [31333, 32726, 32748, 32768, 0],
                [20618, 31487, 32544, 32768, 0],
                [12901, 27217, 31232, 32768, 0],
                [8624, 21734, 28171, 32768, 0],
                [5104, 14191, 20748, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [11206, 21090, 26561, 32768, 0],
                [28759, 32279, 32671, 32768, 0],
                [14171, 27952, 31569, 32768, 0],
                [9743, 22907, 29141, 32768, 0],
                [6871, 17886, 24868, 32768, 0],
                [4960, 13152, 19315, 32768, 0],
                [31077, 32661, 32748, 32768, 0],
                [19400, 31195, 32515, 32768, 0],
                [12752, 26858, 31040, 32768, 0],
                [8370, 22098, 28591, 32768, 0],
                [5457, 15373, 22298, 32768, 0],
                [31697, 32706, 32748, 32768, 0],
                [17860, 30657, 32333, 32768, 0],
                [12510, 24812, 29261, 32768, 0],
                [6180, 19124, 24722, 32768, 0],
                [5041, 13548, 17959, 32768, 0],
                [31552, 32716, 32748, 32768, 0],
                [21908, 31769, 32623, 32768, 0],
                [14470, 28201, 31565, 32768, 0],
                [9493, 22982, 28608, 32768, 0],
                [6858, 17240, 24137, 32768, 0],
                [32543, 32752, 32756, 32768, 0],
                [24286, 32097, 32666, 32768, 0],
                [15958, 29217, 32024, 32768, 0],
                [10207, 24234, 29958, 32768, 0],
                [6929, 18305, 25652, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
        [
            [
                [4137, 10847, 15682, 32768, 0],
                [17824, 27001, 30058, 32768, 0],
                [10204, 22796, 28291, 32768, 0],
                [6076, 15935, 22125, 32768, 0],
                [3852, 10937, 16816, 32768, 0],
                [2252, 6324, 10131, 32768, 0],
                [25840, 32016, 32662, 32768, 0],
                [15109, 28268, 31531, 32768, 0],
                [9385, 22231, 28340, 32768, 0],
                [6082, 16672, 23479, 32768, 0],
                [3318, 9427, 14681, 32768, 0],
                [30594, 32574, 32718, 32768, 0],
                [16836, 29552, 31859, 32768, 0],
                [9556, 22542, 28356, 32768, 0],
                [6305, 16725, 23540, 32768, 0],
                [3376, 9895, 15184, 32768, 0],
                [29383, 32617, 32745, 32768, 0],
                [18891, 30809, 32401, 32768, 0],
                [11688, 25942, 30687, 32768, 0],
                [7468, 19469, 26651, 32768, 0],
                [3909, 11358, 17012, 32768, 0],
                [31564, 32736, 32748, 32768, 0],
                [20906, 31611, 32600, 32768, 0],
                [13191, 27621, 31537, 32768, 0],
                [8768, 22029, 28676, 32768, 0],
                [5079, 14109, 20906, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
];

// ---------------------------------------------------------------------
// Round 140 — coefficient `coeff_br` sub-group default CDF (§9.4), the
// LAST of the `coeff_base` / `coeff_base_eob` / `coeff_br` braid reset
// by the §8.3.1 `init_coeff_cdfs` function. `coeff_br` codes the
// per-coefficient base-range increment used to push a level above
// `NUM_BASE_LEVELS`: each read codes a value in `0..BR_CDF_SIZE` (the
// row is `[c0, c1, c2, 32768, 0]` — three cumulative frequencies plus
// the §8.3 adaptation counter), and §5.11.39 stacks
// `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) == 4` such reads per
// coefficient. Transcribed verbatim from the §9.4 source; the outer
// `COEFF_CDF_Q_CTXS` axis is selected by `base_q_idx` at
// `init_coeff_cdfs`. With this table all three coeff-CDF braids are
// landed; the next gate is the §8.3.2 `get_coeff_base_ctx()` /
// `get_br_ctx()` neighbour-derivation helpers (deferred to a future
// round — they need tile-content walker state).
// ---------------------------------------------------------------------

/// `Default_Coeff_Br_Cdf[ COEFF_CDF_Q_CTXS ][ TX_SIZES ][ PLANE_TYPES ][ LEVEL_CONTEXTS ][ BR_CDF_SIZE + 1 ]`
/// (§9.4). Codes `coeff_br`. Selected at `init_coeff_cdfs` by `idx`
/// (the [`coeff_cdf_q_ctx`] value), then by `Min(txSzCtx, TX_32X32)`,
/// `ptype`, and the `coeff_br` context (the §8.3.2 `get_br_ctx(...)`
/// result, in `0..LEVEL_CONTEXTS`).
///
/// Declared `static` rather than `const` because the table is large
/// (840 5-entry rows = 8400 bytes) and `clippy::large_const_arrays`
/// flags const arrays of this size as a per-use copy hazard. The
/// shape and read semantics are otherwise identical to the other
/// `Default_*_Cdf` arrays in this module.
pub static DEFAULT_COEFF_BR_CDF: [[[[[u16; BR_CDF_SIZE + 1]; LEVEL_CONTEXTS]; PLANE_TYPES];
    TX_SIZES]; COEFF_CDF_Q_CTXS] = [
    [
        [
            [
                [14298, 20718, 24174, 32768, 0],
                [12536, 19601, 23789, 32768, 0],
                [8712, 15051, 19503, 32768, 0],
                [6170, 11327, 15434, 32768, 0],
                [4742, 8926, 12538, 32768, 0],
                [3803, 7317, 10546, 32768, 0],
                [1696, 3317, 4871, 32768, 0],
                [14392, 19951, 22756, 32768, 0],
                [15978, 23218, 26818, 32768, 0],
                [12187, 19474, 23889, 32768, 0],
                [9176, 15640, 20259, 32768, 0],
                [7068, 12655, 17028, 32768, 0],
                [5656, 10442, 14472, 32768, 0],
                [2580, 4992, 7244, 32768, 0],
                [12136, 18049, 21426, 32768, 0],
                [13784, 20721, 24481, 32768, 0],
                [10836, 17621, 21900, 32768, 0],
                [8372, 14444, 18847, 32768, 0],
                [6523, 11779, 16000, 32768, 0],
                [5337, 9898, 13760, 32768, 0],
                [3034, 5860, 8462, 32768, 0],
            ],
            [
                [15967, 22905, 26286, 32768, 0],
                [13534, 20654, 24579, 32768, 0],
                [9504, 16092, 20535, 32768, 0],
                [6975, 12568, 16903, 32768, 0],
                [5364, 10091, 14020, 32768, 0],
                [4357, 8370, 11857, 32768, 0],
                [2506, 4934, 7218, 32768, 0],
                [23032, 28815, 30936, 32768, 0],
                [19540, 26704, 29719, 32768, 0],
                [15158, 22969, 27097, 32768, 0],
                [11408, 18865, 23650, 32768, 0],
                [8885, 15448, 20250, 32768, 0],
                [7108, 12853, 17416, 32768, 0],
                [4231, 8041, 11480, 32768, 0],
                [19823, 26490, 29156, 32768, 0],
                [18890, 25929, 28932, 32768, 0],
                [15660, 23491, 27433, 32768, 0],
                [12147, 19776, 24488, 32768, 0],
                [9728, 16774, 21649, 32768, 0],
                [7919, 14277, 19066, 32768, 0],
                [5440, 10170, 14185, 32768, 0],
            ],
        ],
        [
            [
                [14406, 20862, 24414, 32768, 0],
                [11824, 18907, 23109, 32768, 0],
                [8257, 14393, 18803, 32768, 0],
                [5860, 10747, 14778, 32768, 0],
                [4475, 8486, 11984, 32768, 0],
                [3606, 6954, 10043, 32768, 0],
                [1736, 3410, 5048, 32768, 0],
                [14430, 20046, 22882, 32768, 0],
                [15593, 22899, 26709, 32768, 0],
                [12102, 19368, 23811, 32768, 0],
                [9059, 15584, 20262, 32768, 0],
                [6999, 12603, 17048, 32768, 0],
                [5684, 10497, 14553, 32768, 0],
                [2822, 5438, 7862, 32768, 0],
                [15785, 21585, 24359, 32768, 0],
                [18347, 25229, 28266, 32768, 0],
                [14974, 22487, 26389, 32768, 0],
                [11423, 18681, 23271, 32768, 0],
                [8863, 15350, 20008, 32768, 0],
                [7153, 12852, 17278, 32768, 0],
                [3707, 7036, 9982, 32768, 0],
            ],
            [
                [15460, 21696, 25469, 32768, 0],
                [12170, 19249, 23191, 32768, 0],
                [8723, 15027, 19332, 32768, 0],
                [6428, 11704, 15874, 32768, 0],
                [4922, 9292, 13052, 32768, 0],
                [4139, 7695, 11010, 32768, 0],
                [2291, 4508, 6598, 32768, 0],
                [19856, 26920, 29828, 32768, 0],
                [17923, 25289, 28792, 32768, 0],
                [14278, 21968, 26297, 32768, 0],
                [10910, 18136, 22950, 32768, 0],
                [8423, 14815, 19627, 32768, 0],
                [6771, 12283, 16774, 32768, 0],
                [4074, 7750, 11081, 32768, 0],
                [19852, 26074, 28672, 32768, 0],
                [19371, 26110, 28989, 32768, 0],
                [16265, 23873, 27663, 32768, 0],
                [12758, 20378, 24952, 32768, 0],
                [10095, 17098, 21961, 32768, 0],
                [8250, 14628, 19451, 32768, 0],
                [5205, 9745, 13622, 32768, 0],
            ],
        ],
        [
            [
                [10563, 16233, 19763, 32768, 0],
                [9794, 16022, 19804, 32768, 0],
                [6750, 11945, 15759, 32768, 0],
                [4963, 9186, 12752, 32768, 0],
                [3845, 7435, 10627, 32768, 0],
                [3051, 6085, 8834, 32768, 0],
                [1311, 2596, 3830, 32768, 0],
                [11246, 16404, 19689, 32768, 0],
                [12315, 18911, 22731, 32768, 0],
                [10557, 17095, 21289, 32768, 0],
                [8136, 14006, 18249, 32768, 0],
                [6348, 11474, 15565, 32768, 0],
                [5196, 9655, 13400, 32768, 0],
                [2349, 4526, 6587, 32768, 0],
                [13337, 18730, 21569, 32768, 0],
                [19306, 26071, 28882, 32768, 0],
                [15952, 23540, 27254, 32768, 0],
                [12409, 19934, 24430, 32768, 0],
                [9760, 16706, 21389, 32768, 0],
                [8004, 14220, 18818, 32768, 0],
                [4138, 7794, 10961, 32768, 0],
            ],
            [
                [10870, 16684, 20949, 32768, 0],
                [9664, 15230, 18680, 32768, 0],
                [6886, 12109, 15408, 32768, 0],
                [4825, 8900, 12305, 32768, 0],
                [3630, 7162, 10314, 32768, 0],
                [3036, 6429, 9387, 32768, 0],
                [1671, 3296, 4940, 32768, 0],
                [13819, 19159, 23026, 32768, 0],
                [11984, 19108, 23120, 32768, 0],
                [10690, 17210, 21663, 32768, 0],
                [7984, 14154, 18333, 32768, 0],
                [6868, 12294, 16124, 32768, 0],
                [5274, 8994, 12868, 32768, 0],
                [2988, 5771, 8424, 32768, 0],
                [19736, 26647, 29141, 32768, 0],
                [18933, 26070, 28984, 32768, 0],
                [15779, 23048, 27200, 32768, 0],
                [12638, 20061, 24532, 32768, 0],
                [10692, 17545, 22220, 32768, 0],
                [9217, 15251, 20054, 32768, 0],
                [5078, 9284, 12594, 32768, 0],
            ],
        ],
        [
            [
                [2331, 3662, 5244, 32768, 0],
                [2891, 4771, 6145, 32768, 0],
                [4598, 7623, 9729, 32768, 0],
                [3520, 6845, 9199, 32768, 0],
                [3417, 6119, 9324, 32768, 0],
                [2601, 5412, 7385, 32768, 0],
                [600, 1173, 1744, 32768, 0],
                [7672, 13286, 17469, 32768, 0],
                [4232, 7792, 10793, 32768, 0],
                [2915, 5317, 7397, 32768, 0],
                [2318, 4356, 6152, 32768, 0],
                [2127, 4000, 5554, 32768, 0],
                [1850, 3478, 5275, 32768, 0],
                [977, 1933, 2843, 32768, 0],
                [18280, 24387, 27989, 32768, 0],
                [15852, 22671, 26185, 32768, 0],
                [13845, 20951, 24789, 32768, 0],
                [11055, 17966, 22129, 32768, 0],
                [9138, 15422, 19801, 32768, 0],
                [7454, 13145, 17456, 32768, 0],
                [3370, 6393, 9013, 32768, 0],
            ],
            [
                [5842, 9229, 10838, 32768, 0],
                [2313, 3491, 4276, 32768, 0],
                [2998, 6104, 7496, 32768, 0],
                [2420, 7447, 9868, 32768, 0],
                [3034, 8495, 10923, 32768, 0],
                [4076, 8937, 10975, 32768, 0],
                [1086, 2370, 3299, 32768, 0],
                [9714, 17254, 20444, 32768, 0],
                [8543, 13698, 17123, 32768, 0],
                [4918, 9007, 11910, 32768, 0],
                [4129, 7532, 10553, 32768, 0],
                [2364, 5533, 8058, 32768, 0],
                [1834, 3546, 5563, 32768, 0],
                [1473, 2908, 4133, 32768, 0],
                [15405, 21193, 25619, 32768, 0],
                [15691, 21952, 26561, 32768, 0],
                [12962, 19194, 24165, 32768, 0],
                [10272, 17855, 22129, 32768, 0],
                [8588, 15270, 20718, 32768, 0],
                [8682, 14669, 19500, 32768, 0],
                [4870, 9636, 13205, 32768, 0],
            ],
        ],
        [
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [14995, 21341, 24749, 32768, 0],
                [13158, 20289, 24601, 32768, 0],
                [8941, 15326, 19876, 32768, 0],
                [6297, 11541, 15807, 32768, 0],
                [4817, 9029, 12776, 32768, 0],
                [3731, 7273, 10627, 32768, 0],
                [1847, 3617, 5354, 32768, 0],
                [14472, 19659, 22343, 32768, 0],
                [16806, 24162, 27533, 32768, 0],
                [12900, 20404, 24713, 32768, 0],
                [9411, 16112, 20797, 32768, 0],
                [7056, 12697, 17148, 32768, 0],
                [5544, 10339, 14460, 32768, 0],
                [2954, 5704, 8319, 32768, 0],
                [12464, 18071, 21354, 32768, 0],
                [15482, 22528, 26034, 32768, 0],
                [12070, 19269, 23624, 32768, 0],
                [8953, 15406, 20106, 32768, 0],
                [7027, 12730, 17220, 32768, 0],
                [5887, 10913, 15140, 32768, 0],
                [3793, 7278, 10447, 32768, 0],
            ],
            [
                [15571, 22232, 25749, 32768, 0],
                [14506, 21575, 25374, 32768, 0],
                [10189, 17089, 21569, 32768, 0],
                [7316, 13301, 17915, 32768, 0],
                [5783, 10912, 15190, 32768, 0],
                [4760, 9155, 13088, 32768, 0],
                [2993, 5966, 8774, 32768, 0],
                [23424, 28903, 30778, 32768, 0],
                [20775, 27666, 30290, 32768, 0],
                [16474, 24410, 28299, 32768, 0],
                [12471, 20180, 24987, 32768, 0],
                [9410, 16487, 21439, 32768, 0],
                [7536, 13614, 18529, 32768, 0],
                [5048, 9586, 13549, 32768, 0],
                [21090, 27290, 29756, 32768, 0],
                [20796, 27402, 30026, 32768, 0],
                [17819, 25485, 28969, 32768, 0],
                [13860, 21909, 26462, 32768, 0],
                [11002, 18494, 23529, 32768, 0],
                [8953, 15929, 20897, 32768, 0],
                [6448, 11918, 16454, 32768, 0],
            ],
        ],
        [
            [
                [15999, 22208, 25449, 32768, 0],
                [13050, 19988, 24122, 32768, 0],
                [8594, 14864, 19378, 32768, 0],
                [6033, 11079, 15238, 32768, 0],
                [4554, 8683, 12347, 32768, 0],
                [3672, 7139, 10337, 32768, 0],
                [1900, 3771, 5576, 32768, 0],
                [15788, 21340, 23949, 32768, 0],
                [16825, 24235, 27758, 32768, 0],
                [12873, 20402, 24810, 32768, 0],
                [9590, 16363, 21094, 32768, 0],
                [7352, 13209, 17733, 32768, 0],
                [5960, 10989, 15184, 32768, 0],
                [3232, 6234, 9007, 32768, 0],
                [15761, 20716, 23224, 32768, 0],
                [19318, 25989, 28759, 32768, 0],
                [15529, 23094, 26929, 32768, 0],
                [11662, 18989, 23641, 32768, 0],
                [8955, 15568, 20366, 32768, 0],
                [7281, 13106, 17708, 32768, 0],
                [4248, 8059, 11440, 32768, 0],
            ],
            [
                [14899, 21217, 24503, 32768, 0],
                [13519, 20283, 24047, 32768, 0],
                [9429, 15966, 20365, 32768, 0],
                [6700, 12355, 16652, 32768, 0],
                [5088, 9704, 13716, 32768, 0],
                [4243, 8154, 11731, 32768, 0],
                [2702, 5364, 7861, 32768, 0],
                [22745, 28388, 30454, 32768, 0],
                [20235, 27146, 29922, 32768, 0],
                [15896, 23715, 27637, 32768, 0],
                [11840, 19350, 24131, 32768, 0],
                [9122, 15932, 20880, 32768, 0],
                [7488, 13581, 18362, 32768, 0],
                [5114, 9568, 13370, 32768, 0],
                [20845, 26553, 28932, 32768, 0],
                [20981, 27372, 29884, 32768, 0],
                [17781, 25335, 28785, 32768, 0],
                [13760, 21708, 26297, 32768, 0],
                [10975, 18415, 23365, 32768, 0],
                [9045, 15789, 20686, 32768, 0],
                [6130, 11199, 15423, 32768, 0],
            ],
        ],
        [
            [
                [13549, 19724, 23158, 32768, 0],
                [11844, 18382, 22246, 32768, 0],
                [7919, 13619, 17773, 32768, 0],
                [5486, 10143, 13946, 32768, 0],
                [4166, 7983, 11324, 32768, 0],
                [3364, 6506, 9427, 32768, 0],
                [1598, 3160, 4674, 32768, 0],
                [15281, 20979, 23781, 32768, 0],
                [14939, 22119, 25952, 32768, 0],
                [11363, 18407, 22812, 32768, 0],
                [8609, 14857, 19370, 32768, 0],
                [6737, 12184, 16480, 32768, 0],
                [5506, 10263, 14262, 32768, 0],
                [2990, 5786, 8380, 32768, 0],
                [20249, 25253, 27417, 32768, 0],
                [21070, 27518, 30001, 32768, 0],
                [16854, 24469, 28074, 32768, 0],
                [12864, 20486, 25000, 32768, 0],
                [9962, 16978, 21778, 32768, 0],
                [8074, 14338, 19048, 32768, 0],
                [4494, 8479, 11906, 32768, 0],
            ],
            [
                [13960, 19617, 22829, 32768, 0],
                [11150, 17341, 21228, 32768, 0],
                [7150, 12964, 17190, 32768, 0],
                [5331, 10002, 13867, 32768, 0],
                [4167, 7744, 11057, 32768, 0],
                [3480, 6629, 9646, 32768, 0],
                [1883, 3784, 5686, 32768, 0],
                [18752, 25660, 28912, 32768, 0],
                [16968, 24586, 28030, 32768, 0],
                [13520, 21055, 25313, 32768, 0],
                [10453, 17626, 22280, 32768, 0],
                [8386, 14505, 19116, 32768, 0],
                [6742, 12595, 17008, 32768, 0],
                [4273, 8140, 11499, 32768, 0],
                [22120, 27827, 30233, 32768, 0],
                [20563, 27358, 29895, 32768, 0],
                [17076, 24644, 28153, 32768, 0],
                [13362, 20942, 25309, 32768, 0],
                [10794, 17965, 22695, 32768, 0],
                [9014, 15652, 20319, 32768, 0],
                [5708, 10512, 14497, 32768, 0],
            ],
        ],
        [
            [
                [5705, 10930, 15725, 32768, 0],
                [7946, 12765, 16115, 32768, 0],
                [6801, 12123, 16226, 32768, 0],
                [5462, 10135, 14200, 32768, 0],
                [4189, 8011, 11507, 32768, 0],
                [3191, 6229, 9408, 32768, 0],
                [1057, 2137, 3212, 32768, 0],
                [10018, 17067, 21491, 32768, 0],
                [7380, 12582, 16453, 32768, 0],
                [6068, 10845, 14339, 32768, 0],
                [5098, 9198, 12555, 32768, 0],
                [4312, 8010, 11119, 32768, 0],
                [3700, 6966, 9781, 32768, 0],
                [1693, 3326, 4887, 32768, 0],
                [18757, 24930, 27774, 32768, 0],
                [17648, 24596, 27817, 32768, 0],
                [14707, 22052, 26026, 32768, 0],
                [11720, 18852, 23292, 32768, 0],
                [9357, 15952, 20525, 32768, 0],
                [7810, 13753, 18210, 32768, 0],
                [3879, 7333, 10328, 32768, 0],
            ],
            [
                [8278, 13242, 15922, 32768, 0],
                [10547, 15867, 18919, 32768, 0],
                [9106, 15842, 20609, 32768, 0],
                [6833, 13007, 17218, 32768, 0],
                [4811, 9712, 13923, 32768, 0],
                [3985, 7352, 11128, 32768, 0],
                [1688, 3458, 5262, 32768, 0],
                [12951, 21861, 26510, 32768, 0],
                [9788, 16044, 20276, 32768, 0],
                [6309, 11244, 14870, 32768, 0],
                [5183, 9349, 12566, 32768, 0],
                [4389, 8229, 11492, 32768, 0],
                [3633, 6945, 10620, 32768, 0],
                [3600, 6847, 9907, 32768, 0],
                [21748, 28137, 30255, 32768, 0],
                [19436, 26581, 29560, 32768, 0],
                [16359, 24201, 27953, 32768, 0],
                [13961, 21693, 25871, 32768, 0],
                [11544, 18686, 23322, 32768, 0],
                [9372, 16462, 20952, 32768, 0],
                [6138, 11210, 15390, 32768, 0],
            ],
        ],
        [
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [16138, 22223, 25509, 32768, 0],
                [15347, 22430, 26332, 32768, 0],
                [9614, 16736, 21332, 32768, 0],
                [6600, 12275, 16907, 32768, 0],
                [4811, 9424, 13547, 32768, 0],
                [3748, 7809, 11420, 32768, 0],
                [2254, 4587, 6890, 32768, 0],
                [15196, 20284, 23177, 32768, 0],
                [18317, 25469, 28451, 32768, 0],
                [13918, 21651, 25842, 32768, 0],
                [10052, 17150, 21995, 32768, 0],
                [7499, 13630, 18587, 32768, 0],
                [6158, 11417, 16003, 32768, 0],
                [4014, 7785, 11252, 32768, 0],
                [15048, 21067, 24384, 32768, 0],
                [18202, 25346, 28553, 32768, 0],
                [14302, 22019, 26356, 32768, 0],
                [10839, 18139, 23166, 32768, 0],
                [8715, 15744, 20806, 32768, 0],
                [7536, 13576, 18544, 32768, 0],
                [5413, 10335, 14498, 32768, 0],
            ],
            [
                [17394, 24501, 27895, 32768, 0],
                [15889, 23420, 27185, 32768, 0],
                [11561, 19133, 23870, 32768, 0],
                [8285, 14812, 19844, 32768, 0],
                [6496, 12043, 16550, 32768, 0],
                [4771, 9574, 13677, 32768, 0],
                [3603, 6830, 10144, 32768, 0],
                [21656, 27704, 30200, 32768, 0],
                [21324, 27915, 30511, 32768, 0],
                [17327, 25336, 28997, 32768, 0],
                [13417, 21381, 26033, 32768, 0],
                [10132, 17425, 22338, 32768, 0],
                [8580, 15016, 19633, 32768, 0],
                [5694, 11477, 16411, 32768, 0],
                [24116, 29780, 31450, 32768, 0],
                [23853, 29695, 31591, 32768, 0],
                [20085, 27614, 30428, 32768, 0],
                [15326, 24335, 28575, 32768, 0],
                [11814, 19472, 24810, 32768, 0],
                [10221, 18611, 24767, 32768, 0],
                [7689, 14558, 20321, 32768, 0],
            ],
        ],
        [
            [
                [16214, 22380, 25770, 32768, 0],
                [14213, 21304, 25295, 32768, 0],
                [9213, 15823, 20455, 32768, 0],
                [6395, 11758, 16139, 32768, 0],
                [4779, 9187, 13066, 32768, 0],
                [3821, 7501, 10953, 32768, 0],
                [2293, 4567, 6795, 32768, 0],
                [15859, 21283, 23820, 32768, 0],
                [18404, 25602, 28726, 32768, 0],
                [14325, 21980, 26206, 32768, 0],
                [10669, 17937, 22720, 32768, 0],
                [8297, 14642, 19447, 32768, 0],
                [6746, 12389, 16893, 32768, 0],
                [4324, 8251, 11770, 32768, 0],
                [16532, 21631, 24475, 32768, 0],
                [20667, 27150, 29668, 32768, 0],
                [16728, 24510, 28175, 32768, 0],
                [12861, 20645, 25332, 32768, 0],
                [10076, 17361, 22417, 32768, 0],
                [8395, 14940, 19963, 32768, 0],
                [5731, 10683, 14912, 32768, 0],
            ],
            [
                [14433, 21155, 24938, 32768, 0],
                [14658, 21716, 25545, 32768, 0],
                [9923, 16824, 21557, 32768, 0],
                [6982, 13052, 17721, 32768, 0],
                [5419, 10503, 15050, 32768, 0],
                [4852, 9162, 13014, 32768, 0],
                [3271, 6395, 9630, 32768, 0],
                [22210, 27833, 30109, 32768, 0],
                [20750, 27368, 29821, 32768, 0],
                [16894, 24828, 28573, 32768, 0],
                [13247, 21276, 25757, 32768, 0],
                [10038, 17265, 22563, 32768, 0],
                [8587, 14947, 20327, 32768, 0],
                [5645, 11371, 15252, 32768, 0],
                [22027, 27526, 29714, 32768, 0],
                [23098, 29146, 31221, 32768, 0],
                [19886, 27341, 30272, 32768, 0],
                [15609, 23747, 28046, 32768, 0],
                [11993, 20065, 24939, 32768, 0],
                [9637, 18267, 23671, 32768, 0],
                [7625, 13801, 19144, 32768, 0],
            ],
        ],
        [
            [
                [14438, 20798, 24089, 32768, 0],
                [12621, 19203, 23097, 32768, 0],
                [8177, 14125, 18402, 32768, 0],
                [5674, 10501, 14456, 32768, 0],
                [4236, 8239, 11733, 32768, 0],
                [3447, 6750, 9806, 32768, 0],
                [1986, 3950, 5864, 32768, 0],
                [16208, 22099, 24930, 32768, 0],
                [16537, 24025, 27585, 32768, 0],
                [12780, 20381, 24867, 32768, 0],
                [9767, 16612, 21416, 32768, 0],
                [7686, 13738, 18398, 32768, 0],
                [6333, 11614, 15964, 32768, 0],
                [3941, 7571, 10836, 32768, 0],
                [22819, 27422, 29202, 32768, 0],
                [22224, 28514, 30721, 32768, 0],
                [17660, 25433, 28913, 32768, 0],
                [13574, 21482, 26002, 32768, 0],
                [10629, 17977, 22938, 32768, 0],
                [8612, 15298, 20265, 32768, 0],
                [5607, 10491, 14596, 32768, 0],
            ],
            [
                [13569, 19800, 23206, 32768, 0],
                [13128, 19924, 23869, 32768, 0],
                [8329, 14841, 19403, 32768, 0],
                [6130, 10976, 15057, 32768, 0],
                [4682, 8839, 12518, 32768, 0],
                [3656, 7409, 10588, 32768, 0],
                [2577, 5099, 7412, 32768, 0],
                [22427, 28684, 30585, 32768, 0],
                [20913, 27750, 30139, 32768, 0],
                [15840, 24109, 27834, 32768, 0],
                [12308, 20029, 24569, 32768, 0],
                [10216, 16785, 21458, 32768, 0],
                [8309, 14203, 19113, 32768, 0],
                [6043, 11168, 15307, 32768, 0],
                [23166, 28901, 30998, 32768, 0],
                [21899, 28405, 30751, 32768, 0],
                [18413, 26091, 29443, 32768, 0],
                [15233, 23114, 27352, 32768, 0],
                [12683, 20472, 25288, 32768, 0],
                [10702, 18259, 23409, 32768, 0],
                [8125, 14464, 19226, 32768, 0],
            ],
        ],
        [
            [
                [9040, 14786, 18360, 32768, 0],
                [9979, 15718, 19415, 32768, 0],
                [7913, 13918, 18311, 32768, 0],
                [5859, 10889, 15184, 32768, 0],
                [4593, 8677, 12510, 32768, 0],
                [3820, 7396, 10791, 32768, 0],
                [1730, 3471, 5192, 32768, 0],
                [11803, 18365, 22709, 32768, 0],
                [11419, 18058, 22225, 32768, 0],
                [9418, 15774, 20243, 32768, 0],
                [7539, 13325, 17657, 32768, 0],
                [6233, 11317, 15384, 32768, 0],
                [5137, 9656, 13545, 32768, 0],
                [2977, 5774, 8349, 32768, 0],
                [21207, 27246, 29640, 32768, 0],
                [19547, 26578, 29497, 32768, 0],
                [16169, 23871, 27690, 32768, 0],
                [12820, 20458, 25018, 32768, 0],
                [10224, 17332, 22214, 32768, 0],
                [8526, 15048, 19884, 32768, 0],
                [5037, 9410, 13118, 32768, 0],
            ],
            [
                [12339, 17329, 20140, 32768, 0],
                [13505, 19895, 23225, 32768, 0],
                [9847, 16944, 21564, 32768, 0],
                [7280, 13256, 18348, 32768, 0],
                [4712, 10009, 14454, 32768, 0],
                [4361, 7914, 12477, 32768, 0],
                [2870, 5628, 7995, 32768, 0],
                [20061, 25504, 28526, 32768, 0],
                [15235, 22878, 26145, 32768, 0],
                [12985, 19958, 24155, 32768, 0],
                [9782, 16641, 21403, 32768, 0],
                [9456, 16360, 20760, 32768, 0],
                [6855, 12940, 18557, 32768, 0],
                [5661, 10564, 15002, 32768, 0],
                [25656, 30602, 31894, 32768, 0],
                [22570, 29107, 31092, 32768, 0],
                [18917, 26423, 29541, 32768, 0],
                [15940, 23649, 27754, 32768, 0],
                [12803, 20581, 25219, 32768, 0],
                [11082, 18695, 23376, 32768, 0],
                [7939, 14373, 19005, 32768, 0],
            ],
        ],
        [
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
    [
        [
            [
                [18315, 24289, 27551, 32768, 0],
                [16854, 24068, 27835, 32768, 0],
                [10140, 17927, 23173, 32768, 0],
                [6722, 12982, 18267, 32768, 0],
                [4661, 9826, 14706, 32768, 0],
                [3832, 8165, 12294, 32768, 0],
                [2795, 6098, 9245, 32768, 0],
                [17145, 23326, 26672, 32768, 0],
                [20733, 27680, 30308, 32768, 0],
                [16032, 24461, 28546, 32768, 0],
                [11653, 20093, 25081, 32768, 0],
                [9290, 16429, 22086, 32768, 0],
                [7796, 14598, 19982, 32768, 0],
                [6502, 12378, 17441, 32768, 0],
                [21681, 27732, 30320, 32768, 0],
                [22389, 29044, 31261, 32768, 0],
                [19027, 26731, 30087, 32768, 0],
                [14739, 23755, 28624, 32768, 0],
                [11358, 20778, 25511, 32768, 0],
                [10995, 18073, 24190, 32768, 0],
                [9162, 14990, 20617, 32768, 0],
            ],
            [
                [21425, 27952, 30388, 32768, 0],
                [18062, 25838, 29034, 32768, 0],
                [11956, 19881, 24808, 32768, 0],
                [7718, 15000, 20980, 32768, 0],
                [5702, 11254, 16143, 32768, 0],
                [4898, 9088, 16864, 32768, 0],
                [3679, 6776, 11907, 32768, 0],
                [23294, 30160, 31663, 32768, 0],
                [24397, 29896, 31836, 32768, 0],
                [19245, 27128, 30593, 32768, 0],
                [13202, 19825, 26404, 32768, 0],
                [11578, 19297, 23957, 32768, 0],
                [8073, 13297, 21370, 32768, 0],
                [5461, 10923, 19745, 32768, 0],
                [27367, 30521, 31934, 32768, 0],
                [24904, 30671, 31940, 32768, 0],
                [23075, 28460, 31299, 32768, 0],
                [14400, 23658, 30417, 32768, 0],
                [13885, 23882, 28325, 32768, 0],
                [14746, 22938, 27853, 32768, 0],
                [5461, 16384, 27307, 32768, 0],
            ],
        ],
        [
            [
                [18274, 24813, 27890, 32768, 0],
                [15537, 23149, 27003, 32768, 0],
                [9449, 16740, 21827, 32768, 0],
                [6700, 12498, 17261, 32768, 0],
                [4988, 9866, 14198, 32768, 0],
                [4236, 8147, 11902, 32768, 0],
                [2867, 5860, 8654, 32768, 0],
                [17124, 23171, 26101, 32768, 0],
                [20396, 27477, 30148, 32768, 0],
                [16573, 24629, 28492, 32768, 0],
                [12749, 20846, 25674, 32768, 0],
                [10233, 17878, 22818, 32768, 0],
                [8525, 15332, 20363, 32768, 0],
                [6283, 11632, 16255, 32768, 0],
                [20466, 26511, 29286, 32768, 0],
                [23059, 29174, 31191, 32768, 0],
                [19481, 27263, 30241, 32768, 0],
                [15458, 23631, 28137, 32768, 0],
                [12416, 20608, 25693, 32768, 0],
                [10261, 18011, 23261, 32768, 0],
                [8016, 14655, 19666, 32768, 0],
            ],
            [
                [17616, 24586, 28112, 32768, 0],
                [15809, 23299, 27155, 32768, 0],
                [10767, 18890, 23793, 32768, 0],
                [7727, 14255, 18865, 32768, 0],
                [6129, 11926, 16882, 32768, 0],
                [4482, 9704, 14861, 32768, 0],
                [3277, 7452, 11522, 32768, 0],
                [22956, 28551, 30730, 32768, 0],
                [22724, 28937, 30961, 32768, 0],
                [18467, 26324, 29580, 32768, 0],
                [13234, 20713, 25649, 32768, 0],
                [11181, 17592, 22481, 32768, 0],
                [8291, 18358, 24576, 32768, 0],
                [7568, 11881, 14984, 32768, 0],
                [24948, 29001, 31147, 32768, 0],
                [25674, 30619, 32151, 32768, 0],
                [20841, 26793, 29603, 32768, 0],
                [14669, 24356, 28666, 32768, 0],
                [11334, 23593, 28219, 32768, 0],
                [8922, 14762, 22873, 32768, 0],
                [8301, 13544, 20535, 32768, 0],
            ],
        ],
        [
            [
                [17113, 23733, 27081, 32768, 0],
                [14139, 21406, 25452, 32768, 0],
                [8552, 15002, 19776, 32768, 0],
                [5871, 11120, 15378, 32768, 0],
                [4455, 8616, 12253, 32768, 0],
                [3469, 6910, 10386, 32768, 0],
                [2255, 4553, 6782, 32768, 0],
                [18224, 24376, 27053, 32768, 0],
                [19290, 26710, 29614, 32768, 0],
                [14936, 22991, 27184, 32768, 0],
                [11238, 18951, 23762, 32768, 0],
                [8786, 15617, 20588, 32768, 0],
                [7317, 13228, 18003, 32768, 0],
                [5101, 9512, 13493, 32768, 0],
                [22639, 28222, 30210, 32768, 0],
                [23216, 29331, 31307, 32768, 0],
                [19075, 26762, 29895, 32768, 0],
                [15014, 23113, 27457, 32768, 0],
                [11938, 19857, 24752, 32768, 0],
                [9942, 17280, 22282, 32768, 0],
                [7167, 13144, 17752, 32768, 0],
            ],
            [
                [15820, 22738, 26488, 32768, 0],
                [13530, 20885, 25216, 32768, 0],
                [8395, 15530, 20452, 32768, 0],
                [6574, 12321, 16380, 32768, 0],
                [5353, 10419, 14568, 32768, 0],
                [4613, 8446, 12381, 32768, 0],
                [3440, 7158, 9903, 32768, 0],
                [24247, 29051, 31224, 32768, 0],
                [22118, 28058, 30369, 32768, 0],
                [16498, 24768, 28389, 32768, 0],
                [12920, 21175, 26137, 32768, 0],
                [10730, 18619, 25352, 32768, 0],
                [10187, 16279, 22791, 32768, 0],
                [9310, 14631, 22127, 32768, 0],
                [24970, 30558, 32057, 32768, 0],
                [24801, 29942, 31698, 32768, 0],
                [22432, 28453, 30855, 32768, 0],
                [19054, 25680, 29580, 32768, 0],
                [14392, 23036, 28109, 32768, 0],
                [12495, 20947, 26650, 32768, 0],
                [12442, 20326, 26214, 32768, 0],
            ],
        ],
        [
            [
                [12162, 18785, 22648, 32768, 0],
                [12749, 19697, 23806, 32768, 0],
                [8580, 15297, 20346, 32768, 0],
                [6169, 11749, 16543, 32768, 0],
                [4836, 9391, 13448, 32768, 0],
                [3821, 7711, 11613, 32768, 0],
                [2228, 4601, 7070, 32768, 0],
                [16319, 24725, 28280, 32768, 0],
                [15698, 23277, 27168, 32768, 0],
                [12726, 20368, 25047, 32768, 0],
                [9912, 17015, 21976, 32768, 0],
                [7888, 14220, 19179, 32768, 0],
                [6777, 12284, 17018, 32768, 0],
                [4492, 8590, 12252, 32768, 0],
                [23249, 28904, 30947, 32768, 0],
                [21050, 27908, 30512, 32768, 0],
                [17440, 25340, 28949, 32768, 0],
                [14059, 22018, 26541, 32768, 0],
                [11288, 18903, 23898, 32768, 0],
                [9411, 16342, 21428, 32768, 0],
                [6278, 11588, 15944, 32768, 0],
            ],
            [
                [13981, 20067, 23226, 32768, 0],
                [16922, 23580, 26783, 32768, 0],
                [11005, 19039, 24487, 32768, 0],
                [7389, 14218, 19798, 32768, 0],
                [5598, 11505, 17206, 32768, 0],
                [6090, 11213, 15659, 32768, 0],
                [3820, 7371, 10119, 32768, 0],
                [21082, 26925, 29675, 32768, 0],
                [21262, 28627, 31128, 32768, 0],
                [18392, 26454, 30437, 32768, 0],
                [14870, 22910, 27096, 32768, 0],
                [12620, 19484, 24908, 32768, 0],
                [9290, 16553, 22802, 32768, 0],
                [6668, 14288, 20004, 32768, 0],
                [27704, 31055, 31949, 32768, 0],
                [24709, 29978, 31788, 32768, 0],
                [21668, 29264, 31657, 32768, 0],
                [18295, 26968, 30074, 32768, 0],
                [16399, 24422, 29313, 32768, 0],
                [14347, 23026, 28104, 32768, 0],
                [12370, 19806, 24477, 32768, 0],
            ],
        ],
        [
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
            [
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
                [8192, 16384, 24576, 32768, 0],
            ],
        ],
    ],
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
    // Round 137 — intra-frame transform-type group. §8.3.1 enumerates
    // each as "`IntraTxTypeSet{n}Cdf` is set equal to a copy of
    // `Default_Intra_Tx_Type_Set{n}_Cdf`". Both tables carry an extra
    // `intraDir` axis (`INTRA_MODES`) on top of the `Tx_Size_Sqr` axis
    // already seen in the inter counterparts.
    // -----------------------------------------------------------------
    /// `TileIntraTxTypeSet1Cdf[ INTRA_TX_TYPE_SET1_SIZES ][ INTRA_MODES ]`
    /// (§8.3.1). Selected for `intra_tx_type` when `set == TX_SET_INTRA_1`.
    pub intra_tx_type_set1:
        [[[u16; TX_TYPES_INTRA_SET1 + 1]; INTRA_MODES]; INTRA_TX_TYPE_SET1_SIZES],
    /// `TileIntraTxTypeSet2Cdf[ INTRA_TX_TYPE_SET2_SIZES ][ INTRA_MODES ]`
    /// (§8.3.1). Selected for `intra_tx_type` when `set == TX_SET_INTRA_2`.
    pub intra_tx_type_set2:
        [[[u16; TX_TYPES_INTRA_SET2 + 1]; INTRA_MODES]; INTRA_TX_TYPE_SET2_SIZES],

    // -----------------------------------------------------------------
    // Round 22 — inter-frame interpolation-filter group. §8.3.1
    // enumerates as "`InterpFilterCdf` is set equal to a copy of
    // `Default_Interp_Filter_Cdf`".
    // -----------------------------------------------------------------
    /// `TileInterpFilterCdf[ INTERP_FILTER_CONTEXTS ]` (§8.3.1). Codes
    /// `interp_filter` (the switchable-filter selection).
    pub interp_filter: [[u16; INTERP_FILTERS + 1]; INTERP_FILTER_CONTEXTS],

    // -----------------------------------------------------------------
    // Round 23 — motion-mode group. §8.3.1 enumerates as
    // "`MotionModeCdf` is set to a copy of `Default_Motion_Mode_Cdf`".
    // -----------------------------------------------------------------
    /// `TileMotionModeCdf[ BLOCK_SIZES ]` (§8.3.1). Codes `motion_mode`
    /// (§5.11.x `read_motion_mode`), selected by `MiSize`.
    pub motion_mode: [[u16; MOTION_MODES + 1]; BLOCK_SIZES],

    // -----------------------------------------------------------------
    // Round 24 — compound-prediction group. §8.3.1 enumerates these as
    // "`CompGroupIdxCdf` is set to a copy of `Default_Comp_Group_Idx_Cdf`",
    // "`CompoundIdxCdf` is set to a copy of `Default_Compound_Idx_Cdf`",
    // and "`CompoundTypeCdf` is set to a copy of
    // `Default_Compound_Type_Cdf`".
    // -----------------------------------------------------------------
    /// `TileCompGroupIdxCdf[ COMP_GROUP_IDX_CONTEXTS ]` (§8.3.1). Codes
    /// the binary `comp_group_idx` (§5.11.x `read_compound_type`),
    /// selected by a precomputed `ctx`.
    pub comp_group_idx: [[u16; 3]; COMP_GROUP_IDX_CONTEXTS],
    /// `TileCompoundIdxCdf[ COMPOUND_IDX_CONTEXTS ]` (§8.3.1). Codes the
    /// binary `compound_idx` (§5.11.x `read_compound_type`), selected by
    /// a precomputed `ctx`.
    pub compound_idx: [[u16; 3]; COMPOUND_IDX_CONTEXTS],
    /// `TileCompoundTypeCdf[ BLOCK_SIZES ]` (§8.3.1). Codes the binary
    /// `compound_type` (§5.11.x `read_compound_type`), selected by
    /// `MiSize`.
    pub compound_type: [[u16; COMPOUND_TYPES + 1]; BLOCK_SIZES],

    // -----------------------------------------------------------------
    // Round 134 — inter-frame intra-mode group. §8.3.1 enumerates these
    // as "`YModeCdf` is set to a copy of `Default_Y_Mode_Cdf`",
    // "`UVModeCflNotAllowedCdf` is set to a copy of
    // `Default_Uv_Mode_Cfl_Not_Allowed_Cdf`", and
    // "`UVModeCflAllowedCdf` is set to a copy of
    // `Default_Uv_Mode_Cfl_Allowed_Cdf`".
    // -----------------------------------------------------------------
    /// `TileYModeCdf[ BLOCK_SIZE_GROUPS ]` (§8.3.1). Codes the inter-frame
    /// `y_mode`, selected by `ctx = Size_Group[ MiSize ]`.
    pub y_mode: [[u16; INTRA_MODES + 1]; BLOCK_SIZE_GROUPS],
    /// `TileUVModeCflNotAllowedCdf[ INTRA_MODES ]` (§8.3.1). Codes
    /// `uv_mode` when chroma-from-luma is not allowed, selected by `YMode`.
    pub uv_mode_cfl_not_allowed: [[u16; UV_INTRA_MODES_CFL_NOT_ALLOWED + 1]; INTRA_MODES],
    /// `TileUVModeCflAllowedCdf[ INTRA_MODES ]` (§8.3.1). Codes `uv_mode`
    /// when chroma-from-luma is allowed, selected by `YMode`.
    pub uv_mode_cfl_allowed: [[u16; UV_INTRA_MODES_CFL_ALLOWED + 1]; INTRA_MODES],

    // -----------------------------------------------------------------
    // Round 135 — angle-delta group. §8.3.1 enumerates this as
    // "`AngleDeltaCdf` is set to a copy of `Default_Angle_Delta_Cdf`".
    // -----------------------------------------------------------------
    /// `TileAngleDeltaCdf[ DIRECTIONAL_MODES ]` (§8.3.1). Codes the
    /// `angle_delta_y` / `angle_delta_uv` directional-angle offset,
    /// selected by `YMode - V_PRED` / `UVMode - V_PRED`.
    pub angle_delta: [[u16; (2 * MAX_ANGLE_DELTA + 1) + 1]; DIRECTIONAL_MODES],

    // -----------------------------------------------------------------
    // Round 136 — coefficient-token entry sub-group. §8.3.1
    // `init_coeff_cdfs` first derives `idx` from `base_q_idx`, then sets
    // each of these working arrays to a copy of `Default_*_Cdf[ idx ]`.
    // The working copy therefore *drops* the `COEFF_CDF_Q_CTXS` axis;
    // [`TileCdfContext::init_coeff_cdfs`] performs the per-`idx` copy.
    // `new_from_defaults` seeds them from `idx == 0` so the value is
    // always well-formed before `init_coeff_cdfs` runs.
    // -----------------------------------------------------------------
    /// `TileTxbSkipCdf[ TX_SIZES ][ TXB_SKIP_CONTEXTS ]` (§8.3.1). Codes
    /// the `all_zero` transform-block skip flag.
    pub txb_skip: [[[u16; 3]; TXB_SKIP_CONTEXTS]; TX_SIZES],
    /// `TileEobPt16Cdf[ PLANE_TYPES ][ 2 ]` (§8.3.1). Codes `eob_pt_16`.
    pub eob_pt_16: [[[u16; 6]; 2]; PLANE_TYPES],
    /// `TileEobPt32Cdf[ PLANE_TYPES ][ 2 ]` (§8.3.1). Codes `eob_pt_32`.
    pub eob_pt_32: [[[u16; 7]; 2]; PLANE_TYPES],
    /// `TileEobPt64Cdf[ PLANE_TYPES ][ 2 ]` (§8.3.1). Codes `eob_pt_64`.
    pub eob_pt_64: [[[u16; 8]; 2]; PLANE_TYPES],
    /// `TileEobPt128Cdf[ PLANE_TYPES ][ 2 ]` (§8.3.1). Codes `eob_pt_128`.
    pub eob_pt_128: [[[u16; 9]; 2]; PLANE_TYPES],
    /// `TileEobPt256Cdf[ PLANE_TYPES ][ 2 ]` (§8.3.1). Codes `eob_pt_256`.
    pub eob_pt_256: [[[u16; 10]; 2]; PLANE_TYPES],
    /// `TileEobPt512Cdf[ PLANE_TYPES ]` (§8.3.1). Codes `eob_pt_512`.
    pub eob_pt_512: [[u16; 11]; PLANE_TYPES],
    /// `TileEobPt1024Cdf[ PLANE_TYPES ]` (§8.3.1). Codes `eob_pt_1024`.
    pub eob_pt_1024: [[u16; 12]; PLANE_TYPES],
    /// `TileEobExtraCdf[ TX_SIZES ][ PLANE_TYPES ][ EOB_COEF_CONTEXTS ]`
    /// (§8.3.1). Codes the binary `eob_extra` flag.
    pub eob_extra: [[[[u16; 3]; EOB_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES],
    /// `TileDcSignCdf[ PLANE_TYPES ][ DC_SIGN_CONTEXTS ]` (§8.3.1). Codes
    /// the binary `dc_sign`.
    pub dc_sign: [[[u16; 3]; DC_SIGN_CONTEXTS]; PLANE_TYPES],

    // -----------------------------------------------------------------
    // Round 138 — `coeff_base_eob` sub-group. §8.3.1 enumerates this as
    // "`CoeffBaseEobCdf` is set to a copy of `Default_Coeff_Base_Eob_Cdf
    // [ idx ]`". The working copy drops the `COEFF_CDF_Q_CTXS` axis;
    // [`TileCdfContext::init_coeff_cdfs`] performs the per-`idx` copy.
    // -----------------------------------------------------------------
    /// `TileCoeffBaseEobCdf[ TX_SIZES ][ PLANE_TYPES ][ SIG_COEF_CONTEXTS_EOB ]`
    /// (§8.3.1). Codes `coeff_base_eob`.
    pub coeff_base_eob: [[[[u16; 4]; SIG_COEF_CONTEXTS_EOB]; PLANE_TYPES]; TX_SIZES],

    // -----------------------------------------------------------------
    // Round 139 — `coeff_base` sub-group. §8.3.1 enumerates this as
    // "`CoeffBaseCdf` is set to a copy of `Default_Coeff_Base_Cdf
    // [ idx ]`". The working copy drops the `COEFF_CDF_Q_CTXS` axis;
    // [`TileCdfContext::init_coeff_cdfs`] performs the per-`idx` copy.
    // -----------------------------------------------------------------
    /// `TileCoeffBaseCdf[ TX_SIZES ][ PLANE_TYPES ][ SIG_COEF_CONTEXTS ]`
    /// (§8.3.1). Codes `coeff_base` (the base level of each non-EOB
    /// coefficient, a 4-symbol alphabet).
    pub coeff_base: [[[[u16; 5]; SIG_COEF_CONTEXTS]; PLANE_TYPES]; TX_SIZES],

    // -----------------------------------------------------------------
    // Round 140 — `coeff_br` sub-group. §8.3.1 enumerates this as
    // "`CoeffBrCdf` is set to a copy of `Default_Coeff_Br_Cdf
    // [ idx ]`". The working copy drops the `COEFF_CDF_Q_CTXS` axis;
    // [`TileCdfContext::init_coeff_cdfs`] performs the per-`idx` copy.
    // The §8.3.2 selector clamps `txSzCtx` to `TX_32X32` before
    // indexing into this working copy, but the working copy itself
    // retains all `TX_SIZES` slices (matching the §9.4 source's
    // outer-after-`idx` shape).
    // -----------------------------------------------------------------
    /// `TileCoeffBrCdf[ TX_SIZES ][ PLANE_TYPES ][ LEVEL_CONTEXTS ]`
    /// (§8.3.1). Codes `coeff_br` (the per-coefficient base-range
    /// increment used to push a level above `NUM_BASE_LEVELS`).
    pub coeff_br: [[[[u16; BR_CDF_SIZE + 1]; LEVEL_CONTEXTS]; PLANE_TYPES]; TX_SIZES],

    // -----------------------------------------------------------------
    // Round 143 — inter-intra group. §8.3.1 enumerates these as
    // "`InterIntraCdf` is set to a copy of `Default_Inter_Intra_Cdf`",
    // "`InterIntraModeCdf` is set to a copy of
    // `Default_Inter_Intra_Mode_Cdf`", and "`WedgeInterIntraCdf` is set
    // to a copy of `Default_Wedge_Inter_Intra_Cdf`".
    // -----------------------------------------------------------------
    /// `TileInterIntraCdf[ BLOCK_SIZE_GROUPS - 1 ]` (§8.3.1). Codes the
    /// binary `interintra` flag (§5.11.28 `read_interintra_mode`),
    /// selected by `ctx = Size_Group[ MiSize ] - 1`.
    pub inter_intra: [[u16; 3]; BLOCK_SIZE_GROUPS - 1],
    /// `TileInterIntraModeCdf[ BLOCK_SIZE_GROUPS - 1 ]` (§8.3.1). Codes
    /// `interintra_mode` (§5.11.28 `read_interintra_mode`), selected by
    /// `ctx = Size_Group[ MiSize ] - 1`.
    pub inter_intra_mode: [[u16; INTERINTRA_MODES + 1]; BLOCK_SIZE_GROUPS - 1],
    /// `TileWedgeInterIntraCdf[ BLOCK_SIZES ]` (§8.3.1). Codes the binary
    /// `wedge_interintra` flag (§5.11.28 `read_interintra_mode`),
    /// selected by `MiSize` per the §8.3.2 selection.
    pub wedge_inter_intra: [[u16; 3]; BLOCK_SIZES],

    // -----------------------------------------------------------------
    // Round 144 — wedge-index CDF. §8.3.1 enumerates this as
    // "`WedgeIndexCdf` is set to a copy of `Default_Wedge_Index_Cdf`".
    // -----------------------------------------------------------------
    /// `TileWedgeIndexCdf[ BLOCK_SIZES ]` (§8.3.1). Codes the
    /// `wedge_index ∈ 0..WEDGE_TYPES` element read by §5.11.28
    /// `read_interintra_mode` (the inter-intra wedge sub-branch, when
    /// `wedge_interintra == 1`) and §5.11.29 `read_compound_type` (the
    /// inter-inter `COMPOUND_WEDGE` branch). Selected by `MiSize` per
    /// the §8.3.2 selection (`TileWedgeIndexCdf[ MiSize ]`).
    pub wedge_index: [[u16; WEDGE_TYPES + 1]; BLOCK_SIZES],
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

            // Round 137 — intra-frame transform-type group.
            intra_tx_type_set1: DEFAULT_INTRA_TX_TYPE_SET1_CDF,
            intra_tx_type_set2: DEFAULT_INTRA_TX_TYPE_SET2_CDF,

            // Round 22 — inter-frame interpolation-filter group.
            interp_filter: DEFAULT_INTERP_FILTER_CDF,

            // Round 23 — motion-mode group.
            motion_mode: DEFAULT_MOTION_MODE_CDF,

            // Round 24 — compound-prediction group.
            comp_group_idx: DEFAULT_COMP_GROUP_IDX_CDF,
            compound_idx: DEFAULT_COMPOUND_IDX_CDF,
            compound_type: DEFAULT_COMPOUND_TYPE_CDF,

            // Round 134 — inter-frame intra-mode group.
            y_mode: DEFAULT_Y_MODE_CDF,
            uv_mode_cfl_not_allowed: DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
            uv_mode_cfl_allowed: DEFAULT_UV_MODE_CFL_ALLOWED_CDF,

            // Round 135 — angle-delta group.
            angle_delta: DEFAULT_ANGLE_DELTA_CDF,

            // Round 136 — coefficient-token entry sub-group. Seeded from
            // the `idx == 0` q-context slice; `init_coeff_cdfs` re-selects
            // the slice for the actual `base_q_idx` before tile content.
            txb_skip: DEFAULT_TXB_SKIP_CDF[0],
            eob_pt_16: DEFAULT_EOB_PT_16_CDF[0],
            eob_pt_32: DEFAULT_EOB_PT_32_CDF[0],
            eob_pt_64: DEFAULT_EOB_PT_64_CDF[0],
            eob_pt_128: DEFAULT_EOB_PT_128_CDF[0],
            eob_pt_256: DEFAULT_EOB_PT_256_CDF[0],
            eob_pt_512: DEFAULT_EOB_PT_512_CDF[0],
            eob_pt_1024: DEFAULT_EOB_PT_1024_CDF[0],
            eob_extra: DEFAULT_EOB_EXTRA_CDF[0],
            dc_sign: DEFAULT_DC_SIGN_CDF[0],

            // Round 138 — `coeff_base_eob` sub-group. Seeded from the
            // `idx == 0` q-context slice; `init_coeff_cdfs` re-selects
            // the slice for the actual `base_q_idx` before tile content.
            coeff_base_eob: DEFAULT_COEFF_BASE_EOB_CDF[0],

            // Round 139 — `coeff_base` sub-group. Seeded from the
            // `idx == 0` q-context slice; `init_coeff_cdfs` re-selects
            // the slice for the actual `base_q_idx` before tile content.
            coeff_base: DEFAULT_COEFF_BASE_CDF[0],

            // Round 140 — `coeff_br` sub-group. Seeded from the
            // `idx == 0` q-context slice; `init_coeff_cdfs` re-selects
            // the slice for the actual `base_q_idx` before tile content.
            coeff_br: DEFAULT_COEFF_BR_CDF[0],

            // Round 143 — inter-intra group.
            inter_intra: DEFAULT_INTER_INTRA_CDF,
            inter_intra_mode: DEFAULT_INTER_INTRA_MODE_CDF,
            wedge_inter_intra: DEFAULT_WEDGE_INTER_INTRA_CDF,

            // Round 144 — wedge-index CDF.
            wedge_index: DEFAULT_WEDGE_INDEX_CDF,
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

    /// §8.3.2 `intra_tx_type`: the cdf depends on the §5.11.48
    /// `get_tx_set()` value as follows (spec §8.3.2, "intra_tx_type"):
    ///
    /// * `TX_SET_INTRA_1` ⇒
    ///   `TileIntraTxTypeSet1Cdf[ Tx_Size_Sqr[ txSz ] ][ intraDir ]`
    /// * `TX_SET_INTRA_2` ⇒
    ///   `TileIntraTxTypeSet2Cdf[ Tx_Size_Sqr[ txSz ] ][ intraDir ]`
    ///
    /// `set` is the §5.11.48 `get_tx_set()` result on the
    /// `is_inter == 0` branch (one of `TX_SET_DCTONLY`, `TX_SET_INTRA_1`,
    /// `TX_SET_INTRA_2`); `tx_size_sqr` is `Tx_Size_Sqr[ txSz ]`
    /// (`0..TX_SIZES`, but only the first 2 entries of Set1 / 3 entries
    /// of Set2 are addressable per the §5.11.48 routing); `intra_dir` is
    /// the §8.3.2 `intraDir` axis — `YMode` when `use_filter_intra == 0`
    /// and `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]`
    /// otherwise (see [`FILTER_INTRA_MODE_TO_INTRA_DIR`]).
    ///
    /// Returns `None` for:
    /// * the §5.11.47 `set == TX_SET_DCTONLY` path (where `intra_tx_type`
    ///   is not read — `TxType = DCT_DCT` is forced),
    /// * any out-of-range `set` / `tx_size_sqr` combination that
    ///   §5.11.48 would not actually produce, and
    /// * `intra_dir >= INTRA_MODES` (a caller bug — the §3 enumeration
    ///   bounds `YMode` to `0..INTRA_MODES`).
    pub fn intra_tx_type_cdf(
        &mut self,
        set: u32,
        tx_size_sqr: u32,
        intra_dir: usize,
    ) -> Option<&mut [u16]> {
        if intra_dir >= INTRA_MODES {
            return None;
        }
        let sqr = tx_size_sqr as usize;
        match set {
            TX_SET_INTRA_1 if sqr < INTRA_TX_TYPE_SET1_SIZES => {
                Some(&mut self.intra_tx_type_set1[sqr][intra_dir])
            }
            TX_SET_INTRA_2 if sqr < INTRA_TX_TYPE_SET2_SIZES => {
                Some(&mut self.intra_tx_type_set2[sqr][intra_dir])
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

    /// §8.3.2 `motion_mode`: the cdf is `TileMotionModeCdf[ MiSize ]`,
    /// a straight `0..BLOCK_SIZES` index — no neighbour-context
    /// arithmetic. Returns `None` if `mi_size >= BLOCK_SIZES` (a caller
    /// bug — the §3 enumeration bounds `MiSize`).
    ///
    /// Per the §9.4 note on `Default_Motion_Mode_Cdf`, `MiSize` indices
    /// `0..=2` and `16..=17` are never reached by the §5.11.x
    /// `read_motion_mode` selection (the §5.11.x gate
    /// `Block_Width >= 8 && Block_Height >= 8` excludes those block
    /// sizes), but the selector still returns the in-table row so the
    /// behaviour is purely indexical.
    pub fn motion_mode_cdf(&mut self, mi_size: usize) -> Option<&mut [u16]> {
        if mi_size < BLOCK_SIZES {
            Some(&mut self.motion_mode[mi_size])
        } else {
            None
        }
    }

    /// §8.3.2 `comp_group_idx`: the cdf is `TileCompGroupIdxCdf[ ctx ]`,
    /// with `ctx` computed by the §8.3.2 `comp_group_idx` paragraph
    /// (which combines `CompGroupIdxs` neighbour values, `AvailU` /
    /// `AvailL`, `AboveSingle` / `LeftSingle` and the `ALTREF_FRAME`
    /// reference test, ending with `ctx = Min(5, ctx)` — in
    /// `0..COMP_GROUP_IDX_CONTEXTS`). The neighbour arithmetic belongs
    /// in the (future) tile walk; this selector takes the already-
    /// computed index.
    pub fn comp_group_idx_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.comp_group_idx[ctx]
    }

    /// §8.3.2 `compound_idx`: the cdf is `TileCompoundIdxCdf[ ctx ]`,
    /// with `ctx` computed by the §8.3.2 `compound_idx` paragraph
    /// (which seeds `ctx` from the `get_relative_dist` forward/backward
    /// equality test, then adds the `CompoundIdxs` neighbour values /
    /// `ALTREF_FRAME` test — in `0..COMPOUND_IDX_CONTEXTS`). The
    /// neighbour arithmetic belongs in the (future) tile walk; this
    /// selector takes the already-computed index.
    pub fn compound_idx_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.compound_idx[ctx]
    }

    /// §8.3.2 `compound_type`: the cdf is
    /// `TileCompoundTypeCdf[ MiSize ]`, a straight `0..BLOCK_SIZES`
    /// index — no neighbour-context arithmetic. Returns `None` if
    /// `mi_size >= BLOCK_SIZES` (a caller bug — the §3 enumeration
    /// bounds `MiSize`).
    ///
    /// Per the §9.4 note on `Default_Compound_Type_Cdf`, `MiSize`
    /// indices `0..=2`, `10..=17` and `20..=21` are never reached by the
    /// §5.11.x `read_compound_type` selection (those block sizes do not
    /// satisfy the masked-compound gate), but the selector still returns
    /// the in-table row so the behaviour is purely indexical.
    pub fn compound_type_cdf(&mut self, mi_size: usize) -> Option<&mut [u16]> {
        if mi_size < BLOCK_SIZES {
            Some(&mut self.compound_type[mi_size])
        } else {
            None
        }
    }

    /// §8.3.2 `y_mode`: the cdf is `TileYModeCdf[ ctx ]`, where the §8.3.2
    /// `y_mode` paragraph computes `ctx = Size_Group[ MiSize ]` (a value
    /// in `0..BLOCK_SIZE_GROUPS`). [`size_group`] performs that mapping;
    /// this selector takes the already-computed context. Returns `None`
    /// if `ctx >= BLOCK_SIZE_GROUPS` (a caller bug).
    ///
    /// Distinct from [`Self::intra_frame_y_mode_cdf`], which is the
    /// keyframe / intra-frame variant keyed by `[abovemode][leftmode]`.
    pub fn y_mode_cdf(&mut self, ctx: usize) -> Option<&mut [u16]> {
        if ctx < BLOCK_SIZE_GROUPS {
            Some(&mut self.y_mode[ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `uv_mode`: selects between the two chroma intra-mode CDFs by
    /// the `cfl_allowed` flag derived in the §8.3.2 `uv_mode` paragraph,
    /// then indexes the chosen array by `YMode`:
    ///
    /// * `cfl_allowed == true` → `TileUVModeCflAllowedCdf[ YMode ]`
    ///   (so `UV_CFL_PRED` is a coded value);
    /// * `cfl_allowed == false` → `TileUVModeCflNotAllowedCdf[ YMode ]`.
    ///
    /// The §8.3.2 derivation of `cfl_allowed` (the `Lossless` /
    /// `get_plane_residual_size` / `Max(Block_Width, Block_Height) <= 32`
    /// tests) belongs in the tile walk; this selector takes the resolved
    /// flag plus `YMode`. Returns `None` if `y_mode >= INTRA_MODES`
    /// (a caller bug — the §3 enumeration bounds `YMode`).
    pub fn uv_mode_cdf(&mut self, cfl_allowed: bool, y_mode: usize) -> Option<&mut [u16]> {
        if y_mode >= INTRA_MODES {
            return None;
        }
        if cfl_allowed {
            Some(&mut self.uv_mode_cfl_allowed[y_mode])
        } else {
            Some(&mut self.uv_mode_cfl_not_allowed[y_mode])
        }
    }

    /// §8.3.2 `angle_delta_y` / `angle_delta_uv`: the cdf is
    /// `TileAngleDeltaCdf[ YMode - V_PRED ]` for the luma element and
    /// `TileAngleDeltaCdf[ UVMode - V_PRED ]` for the chroma element. Both
    /// rebase the directional intra mode onto `0..DIRECTIONAL_MODES` by
    /// subtracting `V_PRED`; this selector takes the directional `mode`
    /// (`YMode` for `angle_delta_y`, `UVMode` for `angle_delta_uv`) and
    /// performs the subtraction.
    ///
    /// `angle_delta_y` / `angle_delta_uv` are only read when the mode is
    /// directional (`V_PRED <= mode <= D67_PRED`, i.e. the `is_directional_mode`
    /// gate in the §5.11 mode-info read). Returns `None` if `mode < V_PRED`
    /// or `mode - V_PRED >= DIRECTIONAL_MODES` (a non-directional mode — a
    /// caller bug, since the syntax wouldn't have coded an `angle_delta`).
    pub fn angle_delta_cdf(&mut self, mode: usize) -> Option<&mut [u16]> {
        if mode < V_PRED {
            return None;
        }
        let idx = mode - V_PRED;
        if idx < DIRECTIONAL_MODES {
            Some(&mut self.angle_delta[idx])
        } else {
            None
        }
    }

    // -----------------------------------------------------------------
    // Round 136 — §8.3.1 `init_coeff_cdfs` + §8.3.2 coefficient-entry
    // selectors.
    // -----------------------------------------------------------------

    /// §8.3.1 `init_coeff_cdfs`: select each coefficient-token working CDF
    /// from its `Default_*_Cdf[ idx ]` slice, where `idx` is derived from
    /// `base_q_idx` by [`coeff_cdf_q_ctx`].
    ///
    /// This is the coefficient counterpart to [`Self::new_from_defaults`]
    /// (which performs `init_non_coeff_cdfs`). It is invoked at the start
    /// of every tile's `coeff( )` parsing.
    pub fn init_coeff_cdfs(&mut self, base_q_idx: u8) {
        let idx = coeff_cdf_q_ctx(base_q_idx);
        self.txb_skip = DEFAULT_TXB_SKIP_CDF[idx];
        self.eob_pt_16 = DEFAULT_EOB_PT_16_CDF[idx];
        self.eob_pt_32 = DEFAULT_EOB_PT_32_CDF[idx];
        self.eob_pt_64 = DEFAULT_EOB_PT_64_CDF[idx];
        self.eob_pt_128 = DEFAULT_EOB_PT_128_CDF[idx];
        self.eob_pt_256 = DEFAULT_EOB_PT_256_CDF[idx];
        self.eob_pt_512 = DEFAULT_EOB_PT_512_CDF[idx];
        self.eob_pt_1024 = DEFAULT_EOB_PT_1024_CDF[idx];
        self.eob_extra = DEFAULT_EOB_EXTRA_CDF[idx];
        self.dc_sign = DEFAULT_DC_SIGN_CDF[idx];
        self.coeff_base_eob = DEFAULT_COEFF_BASE_EOB_CDF[idx];
        self.coeff_base = DEFAULT_COEFF_BASE_CDF[idx];
        self.coeff_br = DEFAULT_COEFF_BR_CDF[idx];
    }

    /// §8.3.2 `all_zero`: the cdf is `TileTxbSkipCdf[ txSzCtx ][ ctx ]`,
    /// where `txSzCtx` is the transform-size context (`0..TX_SIZES`) and
    /// `ctx` is the `all_zero` context (`0..TXB_SKIP_CONTEXTS`). Returns
    /// `None` for out-of-range indices (a caller bug).
    pub fn txb_skip_cdf(&mut self, tx_sz_ctx: usize, ctx: usize) -> Option<&mut [u16]> {
        if tx_sz_ctx < TX_SIZES && ctx < TXB_SKIP_CONTEXTS {
            Some(&mut self.txb_skip[tx_sz_ctx][ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_16` .. `eob_pt_256`: the cdf is
    /// `TileEobPt{N}Cdf[ ptype ][ isInter ]`. `eobMultisize` selects which
    /// of the seven EOB-position tables applies (the `8`-class blocks reach
    /// `eob_pt_512` / `eob_pt_1024`, which have no `isInter` axis — see
    /// [`Self::eob_pt_512_cdf`] / [`Self::eob_pt_1024_cdf`]). `ptype` is the
    /// plane type (`0..PLANE_TYPES`); `is_inter` is `0` or `1`.
    pub fn eob_pt_16_cdf(&mut self, ptype: usize, is_inter: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES && is_inter < 2 {
            Some(&mut self.eob_pt_16[ptype][is_inter])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_32`: `TileEobPt32Cdf[ ptype ][ isInter ]`.
    pub fn eob_pt_32_cdf(&mut self, ptype: usize, is_inter: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES && is_inter < 2 {
            Some(&mut self.eob_pt_32[ptype][is_inter])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_64`: `TileEobPt64Cdf[ ptype ][ isInter ]`.
    pub fn eob_pt_64_cdf(&mut self, ptype: usize, is_inter: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES && is_inter < 2 {
            Some(&mut self.eob_pt_64[ptype][is_inter])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_128`: `TileEobPt128Cdf[ ptype ][ isInter ]`.
    pub fn eob_pt_128_cdf(&mut self, ptype: usize, is_inter: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES && is_inter < 2 {
            Some(&mut self.eob_pt_128[ptype][is_inter])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_256`: `TileEobPt256Cdf[ ptype ][ isInter ]`.
    pub fn eob_pt_256_cdf(&mut self, ptype: usize, is_inter: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES && is_inter < 2 {
            Some(&mut self.eob_pt_256[ptype][is_inter])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_512`: `TileEobPt512Cdf[ ptype ]` (no `isInter` axis).
    pub fn eob_pt_512_cdf(&mut self, ptype: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES {
            Some(&mut self.eob_pt_512[ptype])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_pt_1024`: `TileEobPt1024Cdf[ ptype ]` (no `isInter`
    /// axis).
    pub fn eob_pt_1024_cdf(&mut self, ptype: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES {
            Some(&mut self.eob_pt_1024[ptype])
        } else {
            None
        }
    }

    /// §8.3.2 `eob_extra`: the cdf is
    /// `TileEobExtraCdf[ txSzCtx ][ ptype ][ ctx ]`, where `ctx` is the
    /// `eob_extra` context (`0..EOB_COEF_CONTEXTS`). Returns `None` for an
    /// out-of-range index.
    pub fn eob_extra_cdf(
        &mut self,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> Option<&mut [u16]> {
        if tx_sz_ctx < TX_SIZES && ptype < PLANE_TYPES && ctx < EOB_COEF_CONTEXTS {
            Some(&mut self.eob_extra[tx_sz_ctx][ptype][ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `dc_sign`: the cdf is `TileDcSignCdf[ ptype ][ ctx ]`, where
    /// `ctx` is the `dc_sign` context (`0..DC_SIGN_CONTEXTS`). Returns
    /// `None` for an out-of-range index.
    pub fn dc_sign_cdf(&mut self, ptype: usize, ctx: usize) -> Option<&mut [u16]> {
        if ptype < PLANE_TYPES && ctx < DC_SIGN_CONTEXTS {
            Some(&mut self.dc_sign[ptype][ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `coeff_base_eob`: the cdf is
    /// `TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]`, where `ctx` is
    /// the `coeff_base_eob` context derived per §8.3.2 from
    /// `get_coeff_base_ctx(...) - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`
    /// (the `get_coeff_base_ctx()` lookup itself belongs to the
    /// not-yet-implemented tile-content walk). Returns `None` for an
    /// out-of-range index.
    pub fn coeff_base_eob_cdf(
        &mut self,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> Option<&mut [u16]> {
        if tx_sz_ctx < TX_SIZES && ptype < PLANE_TYPES && ctx < SIG_COEF_CONTEXTS_EOB {
            Some(&mut self.coeff_base_eob[tx_sz_ctx][ptype][ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `coeff_base`: the cdf is
    /// `TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]`, where `ctx` is the
    /// `coeff_base` context — the §8.3.2 `get_coeff_base_ctx(...)` result
    /// for a non-EOB coefficient, in `0..SIG_COEF_CONTEXTS` (the
    /// `get_coeff_base_ctx()` lookup itself belongs to the
    /// not-yet-implemented tile-content walk). Returns `None` for an
    /// out-of-range index.
    pub fn coeff_base_cdf(
        &mut self,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> Option<&mut [u16]> {
        if tx_sz_ctx < TX_SIZES && ptype < PLANE_TYPES && ctx < SIG_COEF_CONTEXTS {
            Some(&mut self.coeff_base[tx_sz_ctx][ptype][ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `coeff_br`: the cdf is
    /// `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]`, where
    /// `ctx` is the §8.3.2 `get_br_ctx(...)` result in
    /// `0..LEVEL_CONTEXTS` (the `get_br_ctx()` neighbour-magnitude
    /// derivation itself needs tile-content walker state and belongs to
    /// a later round).
    ///
    /// The §3 `TX_32X32` constant (`= 3`) caps the `txSzCtx` selector at
    /// the largest size for which `coeff_br` is parsed; this function
    /// performs that clamp before indexing. Returns `None` only for an
    /// out-of-range `ptype` / `ctx`; any `tx_sz_ctx` is accepted because
    /// the selector clamps it.
    pub fn coeff_br_cdf(
        &mut self,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> Option<&mut [u16]> {
        // §3 `TX_32X32 = 3`; clamp per the §8.3.2 spec selector.
        const TX_32X32: usize = 3;
        let clamped = if tx_sz_ctx > TX_32X32 {
            TX_32X32
        } else {
            tx_sz_ctx
        };
        if ptype < PLANE_TYPES && ctx < LEVEL_CONTEXTS {
            Some(&mut self.coeff_br[clamped][ptype][ctx])
        } else {
            None
        }
    }

    // -----------------------------------------------------------------
    // Round 143 — §8.3.2 inter-intra selectors.
    // -----------------------------------------------------------------

    /// §8.3.2 `interintra`: the cdf is `TileInterIntraCdf[ ctx ]`, where
    /// the §8.3.2 `interintra` paragraph computes
    /// `ctx = Size_Group[ MiSize ] - 1` (a value in
    /// `0..(BLOCK_SIZE_GROUPS - 1)`). [`interintra_ctx`] performs that
    /// mapping; this selector takes the already-computed context. Returns
    /// `None` if `ctx >= BLOCK_SIZE_GROUPS - 1` (a caller bug).
    pub fn inter_intra_cdf(&mut self, ctx: usize) -> Option<&mut [u16]> {
        if ctx < BLOCK_SIZE_GROUPS - 1 {
            Some(&mut self.inter_intra[ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `interintra_mode`: the cdf is
    /// `TileInterIntraModeCdf[ ctx ]`, where the §8.3.2 `interintra_mode`
    /// paragraph computes `ctx = Size_Group[ MiSize ] - 1` (a value in
    /// `0..(BLOCK_SIZE_GROUPS - 1)`). [`interintra_ctx`] performs that
    /// mapping; this selector takes the already-computed context. Returns
    /// `None` if `ctx >= BLOCK_SIZE_GROUPS - 1` (a caller bug).
    pub fn inter_intra_mode_cdf(&mut self, ctx: usize) -> Option<&mut [u16]> {
        if ctx < BLOCK_SIZE_GROUPS - 1 {
            Some(&mut self.inter_intra_mode[ctx])
        } else {
            None
        }
    }

    /// §8.3.2 `wedge_interintra`: the cdf is
    /// `TileWedgeInterIntraCdf[ MiSize ]`, a straight `0..BLOCK_SIZES`
    /// index. Returns `None` if `mi_size >= BLOCK_SIZES` (a caller bug).
    ///
    /// Per the §9.4 note (and the §5.11.28 syntax gate), only
    /// `BLOCK_8X8..=BLOCK_32X32` (indices 3..=9) are reachable in
    /// practice; the other rows are placeholder values and are surfaced
    /// to keep `MiSize`-indexing uniform.
    pub fn wedge_inter_intra_cdf(&mut self, mi_size: usize) -> Option<&mut [u16]> {
        if mi_size < BLOCK_SIZES {
            Some(&mut self.wedge_inter_intra[mi_size])
        } else {
            None
        }
    }

    /// §8.3.2 `wedge_index`: the cdf is `TileWedgeIndexCdf[ MiSize ]`,
    /// a straight `0..BLOCK_SIZES` index. Returns `None` if
    /// `mi_size >= BLOCK_SIZES` (a caller bug).
    ///
    /// Per the §9.4 `Default_Wedge_Index_Cdf` note ("Indices 0 to 2, 10
    /// to 17, and 20 to 21 inclusive are never used in the first
    /// dimension"), `wedge_index` is only ever coded for block sizes
    /// where `Wedge_Bits[ MiSize ] > 0` (§3 table — indices 3..=9 plus
    /// 18..=19); the other rows are placeholder uniform CDFs and are
    /// surfaced to keep `MiSize`-indexing uniform with every other
    /// `[ BLOCK_SIZES ]` table.
    pub fn wedge_index_cdf(&mut self, mi_size: usize) -> Option<&mut [u16]> {
        if mi_size < BLOCK_SIZES {
            Some(&mut self.wedge_index[mi_size])
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

/// §8.3.2 `y_mode` context mapping: `ctx = Size_Group[ MiSize ]`. The
/// result selects a [`TileCdfContext::y_mode_cdf`] row and is in
/// `0..BLOCK_SIZE_GROUPS`. `mi_size` is the current block's `MiSize`
/// (`0..BLOCK_SIZES`).
pub fn size_group(mi_size: usize) -> usize {
    SIZE_GROUP[mi_size]
}

/// §8.3.1 `init_coeff_cdfs`: derive the coefficient-CDF q-context `idx`
/// from `base_q_idx`:
///
/// ```text
///   if      base_q_idx <= 20  idx = 0
///   else if base_q_idx <= 60  idx = 1
///   else if base_q_idx <= 120 idx = 2
///   else                      idx = 3
/// ```
///
/// The result selects the `Default_*_Cdf[ idx ]` slice copied into the
/// working coefficient CDFs (and is in `0..COEFF_CDF_Q_CTXS`).
pub fn coeff_cdf_q_ctx(base_q_idx: u8) -> usize {
    if base_q_idx <= 20 {
        0
    } else if base_q_idx <= 60 {
        1
    } else if base_q_idx <= 120 {
        2
    } else {
        3
    }
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
/// branch is covered by the companion [`intra_tx_type_set`].
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
// Round 137 — intra-frame transform-type §8.3.2 helpers. Mirrors the
// `is_inter == 0` branch of §5.11.48 `get_tx_set()` and the §8.3.2
// `intraDir` derivation (`YMode` or
// `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]`).
// ---------------------------------------------------------------------

/// §5.11.48 `get_tx_set()` (the `is_inter == 0` branch). Maps the
/// `txSzSqr` / `txSzSqrUp` pair (and the frame's `reduced_tx_set`) to
/// the intra transform-set index `set ∈ { TX_SET_DCTONLY,
/// TX_SET_INTRA_1, TX_SET_INTRA_2 }`.
///
/// ```text
///   if ( txSzSqrUp > TX_32X32 )                  return TX_SET_DCTONLY
///   if ( txSzSqrUp == TX_32X32 )                 return TX_SET_DCTONLY
///   else if ( reduced_tx_set )                   return TX_SET_INTRA_2
///   else if ( txSzSqr == TX_16X16 )              return TX_SET_INTRA_2
///   return TX_SET_INTRA_1
/// ```
///
/// `tx_sz_sqr` and `tx_sz_sqr_up` are `Tx_Size_Sqr[ txSz ]` and
/// `Tx_Size_Sqr_Up[ txSz ]` (`TX_4X4 = 0..TX_64X64 = 4`).
pub fn intra_tx_type_set(tx_sz_sqr: u32, tx_sz_sqr_up: u32, reduced_tx_set: bool) -> u32 {
    // §3 `TX_32X32 = 3` per the spec's per-`TX_*` constant table.
    const TX_16X16: u32 = 2;
    const TX_32X32: u32 = 3;
    // The two §5.11.48 `return TX_SET_DCTONLY` gates merge: the outer
    // pre-`is_inter` guard fires for `txSzSqrUp > TX_32X32`, and the
    // first intra-branch line fires for `txSzSqrUp == TX_32X32`.
    if tx_sz_sqr_up >= TX_32X32 {
        TX_SET_DCTONLY
    } else if reduced_tx_set || tx_sz_sqr == TX_16X16 {
        TX_SET_INTRA_2
    } else {
        TX_SET_INTRA_1
    }
}

/// §8.3.2 `intra_tx_type` — `intraDir` derivation. Mirrors the spec:
///
/// ```text
///   if ( use_filter_intra == 1 )
///       intraDir = Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]
///   else
///       intraDir = YMode
/// ```
///
/// `y_mode` is the current block's `YMode` (`0..INTRA_MODES`);
/// `filter_intra_mode` is the §5.11.x `filter_intra_mode` syntax value
/// (`0..INTRA_FILTER_MODES`) — supply `0` and `use_filter_intra = false`
/// for blocks that didn't read a filter-intra mode.
///
/// Returns `None` if any input is out of the spec-bounded range (a
/// caller bug — `y_mode < INTRA_MODES` and
/// `filter_intra_mode < INTRA_FILTER_MODES` per the §3 enumerations).
pub fn intra_dir(use_filter_intra: bool, y_mode: usize, filter_intra_mode: usize) -> Option<usize> {
    if use_filter_intra {
        FILTER_INTRA_MODE_TO_INTRA_DIR
            .get(filter_intra_mode)
            .copied()
    } else if y_mode < INTRA_MODES {
        Some(y_mode)
    } else {
        None
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

// ---------------------------------------------------------------------
// §8.3.2 coefficient context derivation (`get_coeff_base_ctx` /
// `get_br_ctx`). These compute the per-coefficient `ctx` index that the
// `coeff_base` / `coeff_base_eob` / `coeff_br` selectors above consume.
// They read a coefficient-magnitude array (`Quant`) plus scalar
// transform / position state supplied by the tile-content walk (which
// is implemented separately); they own the §8.3.2 neighbour scan only.
// ---------------------------------------------------------------------

/// §3 `TX_CLASS_2D` — the two-dimensional transform class returned by
/// `get_tx_class()` for transforms with non-identity behaviour on both
/// axes (the default class for `DCT_DCT` and friends).
pub const TX_CLASS_2D: usize = 0;

/// §3 `TX_CLASS_HORIZ` — the transform class for horizontal-only
/// transforms (`H_DCT` / `H_ADST` / `H_FLIPADST`).
pub const TX_CLASS_HORIZ: usize = 1;

/// §3 `TX_CLASS_VERT` — the transform class for vertical-only
/// transforms (`V_DCT` / `V_ADST` / `V_FLIPADST`).
pub const TX_CLASS_VERT: usize = 2;

/// `TX_SIZES_ALL` (§3) — the number of transform sizes including the
/// rectangular variants; the first axis of [`COEFF_BASE_CTX_OFFSET`]
/// and the index space for [`ADJUSTED_TX_SIZE`] / [`TX_WIDTH`] /
/// [`TX_HEIGHT`] / [`TX_WIDTH_LOG2`].
pub const TX_SIZES_ALL: usize = 19;

/// `Adjusted_Tx_Size[ TX_SIZES_ALL ]` (§ Additional tables). Maps each
/// transform size to the size whose dimensions cap the coefficient
/// context scan at 32×32 — `get_coeff_base_ctx()` / `get_br_ctx()` use
/// `bwl`, `width`, `height` derived from this adjusted size while
/// `Coeff_Base_Ctx_Offset` is still indexed by the *original* size.
/// Entries are themselves `TX_SIZES_ALL` indices.
pub const ADJUSTED_TX_SIZE: [usize; TX_SIZES_ALL] = [
    0,  // TX_4X4   -> TX_4X4
    1,  // TX_8X8   -> TX_8X8
    2,  // TX_16X16 -> TX_16X16
    3,  // TX_32X32 -> TX_32X32
    3,  // TX_64X64 -> TX_32X32
    5,  // TX_4X8   -> TX_4X8
    6,  // TX_8X4   -> TX_8X4
    7,  // TX_8X16  -> TX_8X16
    8,  // TX_16X8  -> TX_16X8
    9,  // TX_16X32 -> TX_16X32
    10, // TX_32X16 -> TX_32X16
    3,  // TX_32X64 -> TX_32X32
    3,  // TX_64X32 -> TX_32X32
    13, // TX_4X16  -> TX_4X16
    14, // TX_16X4  -> TX_16X4
    15, // TX_8X32  -> TX_8X32
    16, // TX_32X8  -> TX_32X8
    9,  // TX_16X64 -> TX_16X32
    10, // TX_64X16 -> TX_32X16
];

/// `Tx_Width[ TX_SIZES_ALL ]` (§ Additional tables) — the width in
/// pixels of each transform size.
pub const TX_WIDTH: [usize; TX_SIZES_ALL] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];

/// `Tx_Height[ TX_SIZES_ALL ]` (§ Additional tables) — the height in
/// pixels of each transform size.
pub const TX_HEIGHT: [usize; TX_SIZES_ALL] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

/// `Tx_Width_Log2[ TX_SIZES_ALL ]` (§ Additional tables) — the base-2
/// logarithm of each transform width (the `bwl` shift used to convert a
/// scan position into `(row, col)` within the coefficient array).
pub const TX_WIDTH_LOG2: [usize; TX_SIZES_ALL] =
    [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];

/// `Tx_Size_Sqr_Up[ TX_SIZES_ALL ]` (§ Additional tables). For a
/// transform size `t` (of width `w` and height `h`), returns the square
/// tx size whose side length is `Max(w, h)`. Entries are themselves
/// `TX_SIZES_ALL` indices in the `TX_4X4..TX_64X64` range.
///
/// Used by §5.11.40 `compute_tx_type()` (the `txSzSqrUp > TX_32X32`
/// fallback to `DCT_DCT`), by §5.11.48 `get_tx_set()`, and by the
/// `txfm_split` context formula. Replaces the locally-scoped
/// `TX_32X32 = 3` numeric constants the existing
/// [`inter_tx_type_set`] / [`intra_tx_type_set`] helpers used.
pub const TX_SIZE_SQR_UP: [usize; TX_SIZES_ALL] = [
    0, // TX_4X4   -> TX_4X4
    1, // TX_8X8   -> TX_8X8
    2, // TX_16X16 -> TX_16X16
    3, // TX_32X32 -> TX_32X32
    4, // TX_64X64 -> TX_64X64
    1, // TX_4X8   -> TX_8X8
    1, // TX_8X4   -> TX_8X8
    2, // TX_8X16  -> TX_16X16
    2, // TX_16X8  -> TX_16X16
    3, // TX_16X32 -> TX_32X32
    3, // TX_32X16 -> TX_32X32
    4, // TX_32X64 -> TX_64X64
    4, // TX_64X32 -> TX_64X64
    2, // TX_4X16  -> TX_16X16
    2, // TX_16X4  -> TX_16X16
    3, // TX_8X32  -> TX_32X32
    3, // TX_32X8  -> TX_32X32
    4, // TX_16X64 -> TX_64X64
    4, // TX_64X16 -> TX_64X64
];

/// `Sig_Ref_Diff_Offset[ 3 ][ SIG_REF_DIFF_OFFSET_NUM ][ 2 ]`
/// (§ Additional tables). Per transform class, the
/// `(rowDelta, colDelta)` offsets scanned by `get_coeff_base_ctx()` to
/// accumulate the neighbour magnitude. Indexed `[txClass][idx][0|1]`.
pub const SIG_REF_DIFF_OFFSET: [[[usize; 2]; SIG_REF_DIFF_OFFSET_NUM]; 3] = [
    // TX_CLASS_2D
    [[0, 1], [1, 0], [1, 1], [0, 2], [2, 0]],
    // TX_CLASS_HORIZ
    [[0, 1], [1, 0], [0, 2], [0, 3], [0, 4]],
    // TX_CLASS_VERT
    [[0, 1], [1, 0], [2, 0], [3, 0], [4, 0]],
];

/// `Mag_Ref_Offset_With_Tx_Class[ 3 ][ 3 ][ 2 ]` (§ Additional tables).
/// Per transform class, the three `(rowDelta, colDelta)` offsets scanned
/// by `get_br_ctx()` to accumulate the neighbour base-range magnitude.
/// Indexed `[txClass][idx][0|1]`.
pub const MAG_REF_OFFSET_WITH_TX_CLASS: [[[usize; 2]; 3]; 3] = [
    // TX_CLASS_2D
    [[0, 1], [1, 0], [1, 1]],
    // TX_CLASS_HORIZ
    [[0, 1], [1, 0], [0, 2]],
    // TX_CLASS_VERT
    [[0, 1], [1, 0], [2, 0]],
];

/// `Coeff_Base_Pos_Ctx_Offset[ 3 ]` (§8.3.2) — the offsets added to the
/// magnitude bucket in the one-dimensional (vertical / horizontal)
/// branch of `get_coeff_base_ctx()`. Indexed by `Min(idx, 2)` where
/// `idx` is the row (vertical) or column (horizontal) of the position.
pub const COEFF_BASE_POS_CTX_OFFSET: [usize; 3] = [
    SIG_COEF_CONTEXTS_2D,
    SIG_COEF_CONTEXTS_2D + 5,
    SIG_COEF_CONTEXTS_2D + 10,
];

/// `Coeff_Base_Ctx_Offset[ TX_SIZES_ALL ][ 5 ][ 5 ]` (§8.3.2) — the
/// per-position context offset added in the two-dimensional branch of
/// `get_coeff_base_ctx()`. Indexed `[txSz][Min(row,4)][Min(col,4)]`
/// using the *original* (non-adjusted) transform size.
pub const COEFF_BASE_CTX_OFFSET: [[[usize; 5]; 5]; TX_SIZES_ALL] = [
    // TX_4X4
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    // TX_8X8
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_16X16
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_32X32
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_64X64
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_4X8
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    // TX_8X4
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    // TX_8X16
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_16X8
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    // TX_16X32
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_32X16
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    // TX_32X64
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_64X32
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    // TX_4X16
    [
        [0, 11, 11, 11, 0],
        [11, 11, 11, 11, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [21, 21, 21, 21, 0],
    ],
    // TX_16X4
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [0, 0, 0, 0, 0],
    ],
    // TX_8X32
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_32X8
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
    // TX_16X64
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_64X16
    [
        [0, 16, 6, 6, 21],
        [16, 16, 6, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
        [16, 16, 21, 21, 21],
    ],
];

// ---------------------------------------------------------------------
// Round 142 — §5.11.40 `compute_tx_type()` derivation. Pure plumbing
// the tile-content walker will consume to map a 4×4 (blockX, blockY)
// position into the per-plane transform type. The arithmetic itself is
// independent of the msac decoder — it is a table lookup chain over the
// tx-set / tx-type tables, plus the `TxTypes[y][x]` luma cache the
// walker has already filled in.
// ---------------------------------------------------------------------

/// §6.10.16 `TxSize` ordinals (`TxSize` semantics table). Indices into
/// the `Tx_Size_Sqr_Up` / `Tx_Width` / `Tx_Height` / `Tx_Width_Log2`
/// tables. Previously local `const TX_16X16: u32 = 2;` / `const
/// TX_32X32: u32 = 3;` shadows inside the [`inter_tx_type_set`] /
/// [`intra_tx_type_set`] helpers used these same values; the public
/// constants below let new callers (e.g. [`compute_tx_type`]) reference
/// the same ordinals without re-typing them.
pub const TX_4X4: usize = 0;
/// See [`TX_4X4`] for the §6.10.16 ordinal source.
pub const TX_8X8: usize = 1;
/// See [`TX_4X4`] for the §6.10.16 ordinal source.
pub const TX_16X16: usize = 2;
/// See [`TX_4X4`] for the §6.10.16 ordinal source.
pub const TX_32X32: usize = 3;
/// See [`TX_4X4`] for the §6.10.16 ordinal source.
pub const TX_64X64: usize = 4;

/// §6.10.19 / §3 transform-type ordinals. The spec lists `DCT_DCT = 0`
/// through `H_FLIPADST = 15` in a single enumeration; these are the
/// indices used by the [`MODE_TO_TXFM`] / [`TX_TYPE_IN_SET_INTRA`] /
/// [`TX_TYPE_IN_SET_INTER`] / [`get_tx_class`] tables, and the §5.11.40
/// `DCT_DCT` fallback returned by [`compute_tx_type`] when the
/// transform is too large or the per-plane derivation falls back.
pub const DCT_DCT: usize = 0;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const ADST_DCT: usize = 1;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const DCT_ADST: usize = 2;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const ADST_ADST: usize = 3;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const FLIPADST_DCT: usize = 4;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const DCT_FLIPADST: usize = 5;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const FLIPADST_FLIPADST: usize = 6;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const ADST_FLIPADST: usize = 7;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const FLIPADST_ADST: usize = 8;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const IDTX: usize = 9;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const V_DCT: usize = 10;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const H_DCT: usize = 11;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const V_ADST: usize = 12;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const H_ADST: usize = 13;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const V_FLIPADST: usize = 14;
/// See [`DCT_DCT`] for the §6.10.19 ordinal source.
pub const H_FLIPADST: usize = 15;

/// `TX_SET_TYPES_INTRA` (§3) — the number of `is_inter == 0` rows in
/// [`TX_TYPE_IN_SET_INTRA`] (`TX_SET_DCTONLY`, `TX_SET_INTRA_1`,
/// `TX_SET_INTRA_2`).
pub const TX_SET_TYPES_INTRA: usize = 3;

/// `TX_SET_TYPES_INTER` (§3) — the number of `is_inter == 1` rows in
/// [`TX_TYPE_IN_SET_INTER`] (`TX_SET_DCTONLY`, `TX_SET_INTER_1`,
/// `TX_SET_INTER_2`, `TX_SET_INTER_3`).
pub const TX_SET_TYPES_INTER: usize = 4;

/// `Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]` (§ Additional tables).
/// Maps each chroma intra prediction mode to the transform type the
/// §5.11.40 `compute_tx_type()` derivation uses as the per-plane default
/// before the `is_tx_type_in_set` admission filter. Indexed by
/// `UVMode` in `0..UV_INTRA_MODES_CFL_ALLOWED`.
pub const MODE_TO_TXFM: [usize; UV_INTRA_MODES_CFL_ALLOWED] = [
    DCT_DCT,   // DC_PRED
    ADST_DCT,  // V_PRED
    DCT_ADST,  // H_PRED
    DCT_DCT,   // D45_PRED
    ADST_ADST, // D135_PRED
    ADST_DCT,  // D113_PRED
    DCT_ADST,  // D157_PRED
    DCT_ADST,  // D203_PRED
    ADST_DCT,  // D67_PRED
    ADST_ADST, // SMOOTH_PRED
    ADST_DCT,  // SMOOTH_V_PRED
    DCT_ADST,  // SMOOTH_H_PRED
    ADST_ADST, // PAETH_PRED
    DCT_DCT,   // UV_CFL_PRED
];

/// `Tx_Type_In_Set_Intra[ TX_SET_TYPES_INTRA ][ TX_TYPES ]` (§5.11.40).
/// One row per intra `txSet` index (`TX_SET_DCTONLY`, `TX_SET_INTRA_1`,
/// `TX_SET_INTRA_2`). Row `txSet`'s column `txType` is `1` iff the spec
/// considers `txType` admissible for that set; the §5.11.40
/// `is_tx_type_in_set(txSet, txType)` predicate is a direct table read.
pub const TX_TYPE_IN_SET_INTRA: [[u8; TX_TYPES]; TX_SET_TYPES_INTRA] = [
    // TX_SET_DCTONLY: only DCT_DCT.
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // TX_SET_INTRA_1: DCT/ADST cross-set plus IDTX/V_DCT/H_DCT.
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    // TX_SET_INTRA_2: DCT/ADST cross-set plus IDTX.
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
];

/// `Tx_Type_In_Set_Inter[ TX_SET_TYPES_INTER ][ TX_TYPES ]` (§5.11.40).
/// One row per inter `txSet` index (`TX_SET_DCTONLY`, `TX_SET_INTER_1`,
/// `TX_SET_INTER_2`, `TX_SET_INTER_3`). Row `txSet`'s column `txType`
/// is `1` iff the spec considers `txType` admissible for that set; the
/// §5.11.40 `is_tx_type_in_set(txSet, txType)` predicate on the
/// `is_inter == 1` branch is a direct table read.
pub const TX_TYPE_IN_SET_INTER: [[u8; TX_TYPES]; TX_SET_TYPES_INTER] = [
    // TX_SET_DCTONLY: only DCT_DCT.
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // TX_SET_INTER_1: every transform type.
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    // TX_SET_INTER_2: full set minus the four V_/H_FLIPADST tail.
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    // TX_SET_INTER_3: { IDTX, DCT_DCT }.
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
];

/// §5.11.40 `is_tx_type_in_set( txSet, txType )` — the admission filter
/// the [`compute_tx_type`] derivation runs after looking up the
/// per-plane default `txType`. Reads the [`TX_TYPE_IN_SET_INTER`] /
/// [`TX_TYPE_IN_SET_INTRA`] table per the `is_inter` flag and returns
/// `true` when the entry is non-zero.
///
/// Returns `false` for any out-of-range `tx_set` / `tx_type` index
/// rather than panicking; the spec's reachable values always stay in
/// range, so a `false` return tells the caller a bookkeeping bug has
/// presented an unreachable combination.
pub fn is_tx_type_in_set(is_inter: bool, tx_set: u32, tx_type: usize) -> bool {
    let set = tx_set as usize;
    if tx_type >= TX_TYPES {
        return false;
    }
    if is_inter {
        TX_TYPE_IN_SET_INTER
            .get(set)
            .map(|row| row[tx_type] != 0)
            .unwrap_or(false)
    } else {
        TX_TYPE_IN_SET_INTRA
            .get(set)
            .map(|row| row[tx_type] != 0)
            .unwrap_or(false)
    }
}

/// §5.11.40 `compute_tx_type( plane, txSz, blockX, blockY )` — the
/// per-plane transform-type derivation the tile-content walker reads
/// before kicking off coefficient decoding and inverse transform.
///
/// ```text
///   txSzSqrUp = Tx_Size_Sqr_Up[ txSz ]
///   if ( Lossless || txSzSqrUp > TX_32X32 )  return DCT_DCT
///   txSet = get_tx_set( txSz )
///   if ( plane == 0 )                       return TxTypes[ blockY ][ blockX ]
///   if ( is_inter ) {
///       x4 = Max( MiCol, blockX << subsampling_x )
///       y4 = Max( MiRow, blockY << subsampling_y )
///       txType = TxTypes[ y4 ][ x4 ]
///       if ( !is_tx_type_in_set( txSet, txType ) )  return DCT_DCT
///       return txType
///   }
///   txType = Mode_To_Txfm[ UVMode ]
///   if ( !is_tx_type_in_set( txSet, txType ) )      return DCT_DCT
///   return txType
/// ```
///
/// All bitstream-derived state the spec function reads from globals is
/// passed in explicitly:
///
/// * `plane` — `0` for luma (Y), nonzero for chroma (U / V). Only the
///   `plane == 0` branch is special-cased; the §5.11.40 lookup returns
///   the same value for plane 1 and plane 2 with chroma inputs.
/// * `tx_sz` — `TxSize` index in `TX_4X4..TX_64X64..TX_64X16`
///   (§6.10.16 ordinal). Out-of-range inputs return `DCT_DCT` (the
///   §5.11.40 `txSzSqrUp > TX_32X32` fallback's safest extension to a
///   bookkeeping bug).
/// * `lossless`, `is_inter`, `mi_row`, `mi_col`, `subsampling_x` /
///   `subsampling_y`, `uv_mode` — the spec globals of the same names.
/// * `block_x`, `block_y` — the §5.11.40 4×4-granularity block
///   coordinates (in 4×4 luma units; the spec's `blockX << subsampling_x`
///   product yields the same units as `MiCol`).
/// * `tx_types(y, x)` — the §5.11.40 `TxTypes[y][x]` luma cache the
///   walker maintains. Passed as a closure so the helper does not bake
///   in a particular storage shape — the walker may back it by a dense
///   2D array, a sparse map, or a `MiRow / MiCol`-relative tile-local
///   view. The closure is only invoked on the luma / inter chroma
///   branches.
///
/// Returns a transform-type ordinal in `0..TX_TYPES` (see [`DCT_DCT`]
/// through [`H_FLIPADST`]).
#[allow(clippy::too_many_arguments)]
pub fn compute_tx_type<F>(
    plane: usize,
    tx_sz: usize,
    lossless: bool,
    is_inter: bool,
    tx_set: u32,
    mi_row: u32,
    mi_col: u32,
    block_x: u32,
    block_y: u32,
    subsampling_x: u32,
    subsampling_y: u32,
    uv_mode: usize,
    tx_types: F,
) -> usize
where
    F: Fn(u32, u32) -> usize,
{
    // `Lossless || txSzSqrUp > TX_32X32` short-circuit. Treat
    // out-of-range tx_sz as DCT_DCT for the same reason: the spec's
    // reachable inputs all land in `TX_SIZES_ALL`, so an out-of-range
    // index is a caller bug whose safest behaviour matches the
    // §5.11.40 too-large fallback.
    if lossless {
        return DCT_DCT;
    }
    let tx_sz_sqr_up = match TX_SIZE_SQR_UP.get(tx_sz) {
        Some(v) => *v,
        None => return DCT_DCT,
    };
    if tx_sz_sqr_up > TX_32X32 {
        return DCT_DCT;
    }

    if plane == 0 {
        return tx_types(block_y, block_x);
    }

    if is_inter {
        // `x4 = Max( MiCol, blockX << subsampling_x )` —
        // `block_x << subsampling_x` lifts the chroma 4×4 coordinate
        // back into luma units; the `Max` against `MiCol` clips the
        // first row/col of a chroma block to the top-left of its
        // luma siblings (this matters when the chroma block straddles
        // the MI grid origin).
        let x4 = (block_x << subsampling_x).max(mi_col);
        let y4 = (block_y << subsampling_y).max(mi_row);
        let tx_type = tx_types(y4, x4);
        if !is_tx_type_in_set(true, tx_set, tx_type) {
            return DCT_DCT;
        }
        return tx_type;
    }

    let tx_type = match MODE_TO_TXFM.get(uv_mode) {
        Some(v) => *v,
        None => return DCT_DCT,
    };
    if !is_tx_type_in_set(false, tx_set, tx_type) {
        return DCT_DCT;
    }
    tx_type
}

/// §8.3.2 `get_tx_class( txType )` — maps a transform type to its
/// transform class. The vertical-only types (`V_DCT` / `V_ADST` /
/// `V_FLIPADST`) return [`TX_CLASS_VERT`]; the horizontal-only types
/// (`H_DCT` / `H_ADST` / `H_FLIPADST`) return [`TX_CLASS_HORIZ`]; every
/// other transform type returns [`TX_CLASS_2D`].
///
/// The transform-type enumeration values are the §3 `*_DCT` / `*_ADST`
/// / `*_FLIPADST` constants; `v_dct` / `v_adst` / `v_flipadst` and
/// `h_dct` / `h_adst` / `h_flipadst` flag the directional types. The
/// full §8.3.2 `compute_tx_type()` derivation that produces `txType`
/// from the bitstream belongs to the tile-content walk; this helper
/// performs only the class reduction.
pub fn get_tx_class(
    v_dct: bool,
    v_adst: bool,
    v_flipadst: bool,
    h_dct: bool,
    h_adst: bool,
    h_flipadst: bool,
) -> usize {
    if v_dct || v_adst || v_flipadst {
        TX_CLASS_VERT
    } else if h_dct || h_adst || h_flipadst {
        TX_CLASS_HORIZ
    } else {
        TX_CLASS_2D
    }
}

/// §8.3.2 `get_coeff_base_ctx( txSz, plane, blockX, blockY, pos, c,
/// isEob )` — the `coeff_base` / `coeff_base_eob` neighbour-derivation.
///
/// ```text
///   adjTxSz = Adjusted_Tx_Size[ txSz ]
///   bwl     = Tx_Width_Log2[ adjTxSz ]
///   width   = 1 << bwl
///   height  = Tx_Height[ adjTxSz ]
///   if ( isEob ) {                       // EOB-position buckets
///       if ( c == 0 )                  return SIG_COEF_CONTEXTS - 4
///       if ( c <= (height<<bwl)/8 )    return SIG_COEF_CONTEXTS - 3
///       if ( c <= (height<<bwl)/4 )    return SIG_COEF_CONTEXTS - 2
///                                      return SIG_COEF_CONTEXTS - 1
///   }
///   row = pos >> bwl ;  col = pos - (row << bwl)
///   mag = sum over SIG_REF_DIFF_OFFSET_NUM neighbours of
///         Min( Abs( Quant[(refRow<<bwl)+refCol] ), 3 )      // in-bounds only
///   ctx = Min( (mag + 1) >> 1, 4 )
///   if ( txClass == TX_CLASS_2D ) {
///       if ( row == 0 && col == 0 )  return 0
///       return ctx + Coeff_Base_Ctx_Offset[ txSz ][ Min(row,4) ][ Min(col,4) ]
///   }
///   idx = (txClass == TX_CLASS_VERT) ? row : col
///   return ctx + Coeff_Base_Pos_Ctx_Offset[ Min(idx, 2) ]
/// ```
///
/// The neighbour scan uses [`SIG_REF_DIFF_OFFSET`] with the bound check
/// `refRow < height && refCol < width` (`width = 1 << bwl`); the
/// `Coeff_Base_Ctx_Offset` lookup uses the *original* `tx_size`, not the
/// adjusted size. `quant` is the coefficient-magnitude array (`Quant`)
/// indexed `(refRow << bwl) + refCol`; `tx_class` is the
/// [`get_tx_class`] result supplied by the caller (the
/// `compute_tx_type()` derivation belongs to the tile walk); `pos` is
/// the scan position `scan[c]`; `c` is the scan index; `is_eob` selects
/// the EOB-position buckets.
///
/// Returns a value in `0..SIG_COEF_CONTEXTS` (the EOB buckets occupy the
/// top four). The `coeff_base_eob` selector applies
/// `- SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB` to map onto
/// `0..SIG_COEF_CONTEXTS_EOB` — see [`get_coeff_base_eob_ctx`].
pub fn get_coeff_base_ctx(
    quant: &[i32],
    tx_size: usize,
    tx_class: usize,
    pos: usize,
    c: usize,
    is_eob: bool,
) -> usize {
    let adj_tx_sz = ADJUSTED_TX_SIZE[tx_size];
    let bwl = TX_WIDTH_LOG2[adj_tx_sz];
    let width = 1usize << bwl;
    let height = TX_HEIGHT[adj_tx_sz];

    if is_eob {
        if c == 0 {
            return SIG_COEF_CONTEXTS - 4;
        }
        if c <= (height << bwl) / 8 {
            return SIG_COEF_CONTEXTS - 3;
        }
        if c <= (height << bwl) / 4 {
            return SIG_COEF_CONTEXTS - 2;
        }
        return SIG_COEF_CONTEXTS - 1;
    }

    let row = pos >> bwl;
    let col = pos - (row << bwl);

    // Accumulate the neighbour magnitude over the SIG_REF_DIFF_OFFSET_NUM
    // offsets, clamping each in-bounds neighbour magnitude to 3.
    let mut mag: i32 = 0;
    for offset in &SIG_REF_DIFF_OFFSET[tx_class] {
        let ref_row = row + offset[0];
        let ref_col = col + offset[1];
        // The offsets are non-negative, so refRow / refCol >= 0 holds by
        // construction; only the upper bound needs checking.
        if ref_row < height && ref_col < width {
            let q = quant[(ref_row << bwl) + ref_col].unsigned_abs() as i32;
            mag += q.min(3);
        }
    }

    let ctx = (((mag + 1) >> 1).min(4)) as usize;

    if tx_class == TX_CLASS_2D {
        if row == 0 && col == 0 {
            return 0;
        }
        return ctx + COEFF_BASE_CTX_OFFSET[tx_size][row.min(4)][col.min(4)];
    }
    let idx = if tx_class == TX_CLASS_VERT { row } else { col };
    ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)]
}

/// §8.3.2 `coeff_base_eob` context: the [`get_coeff_base_ctx`] result
/// with `is_eob = true`, reduced onto `0..SIG_COEF_CONTEXTS_EOB` via
/// `ctx - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`. The result is the
/// `ctx` index consumed by [`TileCdfContext::coeff_base_eob_cdf`].
pub fn get_coeff_base_eob_ctx(
    quant: &[i32],
    tx_size: usize,
    tx_class: usize,
    pos: usize,
    c: usize,
) -> usize {
    // The §8.3.2 reduction is `ctx - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`
    // in signed arithmetic. The isEob path returns a value in
    // `SIG_COEF_CONTEXTS-4 ..= SIG_COEF_CONTEXTS-1`, so adding the EOB
    // context count first keeps the intermediate non-negative for usize.
    (get_coeff_base_ctx(quant, tx_size, tx_class, pos, c, true) + SIG_COEF_CONTEXTS_EOB)
        - SIG_COEF_CONTEXTS
}

/// §8.3.2 `get_br_ctx` — the `coeff_br` neighbour-derivation.
///
/// ```text
///   adjTxSz = Adjusted_Tx_Size[ txSz ]
///   bwl = Tx_Width_Log2[ adjTxSz ] ; txw = Tx_Width[ adjTxSz ]
///   txh = Tx_Height[ adjTxSz ]
///   row = pos >> bwl ; col = pos - (row << bwl)
///   mag = sum over 3 neighbours of
///         Min( Quant[refRow*txw+refCol], COEFF_BASE_RANGE+NUM_BASE_LEVELS+1 )
///   mag = Min( (mag + 1) >> 1, 6 )
///   if ( pos == 0 )                       ctx = mag
///   else if ( txClass == 0 )              ctx = mag + (row<2 && col<2 ? 7 : 14)
///   else if ( txClass == 1 )              ctx = mag + (col==0 ? 7 : 14)
///   else                                  ctx = mag + (row==0 ? 7 : 14)
/// ```
///
/// The neighbour scan uses [`MAG_REF_OFFSET_WITH_TX_CLASS`] (three
/// offsets) with the bound check `refRow < txh && refCol < (1 << bwl)`.
/// Note the magnitude is `Quant[..]` directly (not its absolute value)
/// and the clamp is `COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1`, distinct
/// from the `get_coeff_base_ctx()` scan. `quant` is the
/// coefficient-magnitude array indexed `refRow * txw + refCol`;
/// `tx_class` is the [`get_tx_class`] result supplied by the caller;
/// `pos` is the scan position.
///
/// Returns a value in `0..LEVEL_CONTEXTS`, the `ctx` index consumed by
/// [`TileCdfContext::coeff_br_cdf`].
pub fn get_br_ctx(quant: &[i32], tx_size: usize, tx_class: usize, pos: usize) -> usize {
    let adj_tx_sz = ADJUSTED_TX_SIZE[tx_size];
    let bwl = TX_WIDTH_LOG2[adj_tx_sz];
    let txw = TX_WIDTH[adj_tx_sz];
    let txh = TX_HEIGHT[adj_tx_sz];
    let row = pos >> bwl;
    let col = pos - (row << bwl);

    let clamp = (COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1) as i32;
    let mut mag: i32 = 0;
    for offset in &MAG_REF_OFFSET_WITH_TX_CLASS[tx_class] {
        let ref_row = row + offset[0];
        let ref_col = col + offset[1];
        if ref_row < txh && ref_col < (1usize << bwl) {
            mag += quant[ref_row * txw + ref_col].min(clamp);
        }
    }

    let mag = (((mag + 1) >> 1).min(6)) as usize;
    if pos == 0 {
        mag
    } else if tx_class == TX_CLASS_2D {
        if row < 2 && col < 2 {
            mag + 7
        } else {
            mag + 14
        }
    } else if tx_class == TX_CLASS_HORIZ {
        if col == 0 {
            mag + 7
        } else {
            mag + 14
        }
    } else if row == 0 {
        mag + 7
    } else {
        mag + 14
    }
}

// ---------------------------------------------------------------------
// Round 143 — inter-intra group (§9.4): `Default_Inter_Intra_Cdf`,
// `Default_Inter_Intra_Mode_Cdf`, `Default_Wedge_Inter_Intra_Cdf`. All
// three CDFs are read by the §5.11.28 `read_interintra_mode` syntax
// inside the inter-block mode-info parse (the `interintra`,
// `interintra_mode`, and `wedge_interintra` elements). §6.10.27 enumerates
// `INTERINTRA_MODES = 4` values for `interintra_mode` (`II_DC_PRED`,
// `II_V_PRED`, `II_H_PRED`, `II_SMOOTH_PRED`).
//
// §8.3.2 selection for `interintra` / `interintra_mode` is
// `ctx = Size_Group[ MiSize ] - 1`; the §5.11.28 gate confines coded
// blocks to the `BLOCK_8X8`..`BLOCK_32X32` band, so the resulting `ctx`
// is one of `{0, 1, 2}` — matching the `[ BLOCK_SIZE_GROUPS - 1 ]` outer
// dimension. The §8.3.2 `wedge_interintra` selection is a straight
// `TileWedgeInterIntraCdf[ MiSize ]` index; per the §9.4 note only block
// sizes 3..=9 (the BLOCK_8X8..BLOCK_32X32 band) are reachable, but the
// table is transcribed full-width to keep the indexing uniform with
// every other `[ BLOCK_SIZES ]` table.
// ---------------------------------------------------------------------

/// `Default_Inter_Intra_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ 3 ]` (§9.4). Binary
/// symbol `interintra` (§5.11.28 `read_interintra_mode` gate; selects
/// whether an inter prediction should be blended with an intra prediction,
/// per §6.10.27 semantics). The §8.3.2 selection computes
/// `ctx = Size_Group[ MiSize ] - 1` (the spec's literal text), valid in
/// `0..(BLOCK_SIZE_GROUPS - 1)`. Each row carries two cumulative
/// frequencies plus the §8.3 adaptation counter (starts at 0).
pub const DEFAULT_INTER_INTRA_CDF: [[u16; 3]; BLOCK_SIZE_GROUPS - 1] =
    [[26887, 32768, 0], [27597, 32768, 0], [30237, 32768, 0]];

/// `Default_Inter_Intra_Mode_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ INTERINTRA_MODES + 1 ]`
/// (§9.4). Codes `interintra_mode ∈ { II_DC_PRED, II_V_PRED, II_H_PRED,
/// II_SMOOTH_PRED }` (§6.10.27 enumeration). The §8.3.2 selection
/// computes `ctx = Size_Group[ MiSize ] - 1`, valid in
/// `0..(BLOCK_SIZE_GROUPS - 1)`. Each row carries
/// `INTERINTRA_MODES = 4` cumulative frequencies plus the §8.3 adaptation
/// counter (starts at 0).
pub const DEFAULT_INTER_INTRA_MODE_CDF: [[u16; INTERINTRA_MODES + 1]; BLOCK_SIZE_GROUPS - 1] = [
    [1875, 11082, 27332, 32768, 0],
    [2473, 9996, 26388, 32768, 0],
    [4238, 11537, 25926, 32768, 0],
];

/// `Default_Wedge_Inter_Intra_Cdf[ BLOCK_SIZES ][ 3 ]` (§9.4). Binary
/// symbol `wedge_interintra` (§5.11.28; §6.10.27 selects whether wedge
/// blending should be used). The §8.3.2 selection is a straight
/// `TileWedgeInterIntraCdf[ MiSize ]` index. Per the §9.4 note only
/// indices 3..=9 (the `BLOCK_8X8`..`BLOCK_32X32` band — the same band
/// the §5.11.28 syntax gate confines coded blocks to) are ever used in
/// the first dimension; the rows outside that band are placeholder
/// `{16384, 32768, 0}` entries and are transcribed full-width to keep
/// `MiSize`-indexing uniform with every other `[ BLOCK_SIZES ]` table.
pub const DEFAULT_WEDGE_INTER_INTRA_CDF: [[u16; 3]; BLOCK_SIZES] = [
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [20036, 32768, 0],
    [24957, 32768, 0],
    [26704, 32768, 0],
    [27530, 32768, 0],
    [29564, 32768, 0],
    [29444, 32768, 0],
    [26872, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
];

/// §8.3.2 `interintra` / `interintra_mode` context mapping:
/// `ctx = Size_Group[ MiSize ] - 1`. The result selects a
/// [`TileCdfContext::inter_intra_cdf`] /
/// [`TileCdfContext::inter_intra_mode_cdf`] row and is in
/// `0..(BLOCK_SIZE_GROUPS - 1)`. `mi_size` is the current block's `MiSize`
/// (`0..BLOCK_SIZES`).
///
/// The §5.11.28 syntax gate restricts coded blocks to
/// `BLOCK_8X8 <= MiSize <= BLOCK_32X32` (per §6.10.27), where
/// `Size_Group[ MiSize ]` is one of `{1, 2, 3}` and the subtraction is
/// well-defined. Returns `None` if `Size_Group[ MiSize ] == 0` (a caller
/// bug: the §5.11.28 gate would not have coded `interintra` /
/// `interintra_mode` in that case) or if `mi_size >= BLOCK_SIZES`.
pub fn interintra_ctx(mi_size: usize) -> Option<usize> {
    if mi_size >= BLOCK_SIZES {
        return None;
    }
    let g = SIZE_GROUP[mi_size];
    if g == 0 {
        None
    } else {
        Some(g - 1)
    }
}

// ---------------------------------------------------------------------
// Round 144 — wedge-index CDF (§9.4): `Default_Wedge_Index_Cdf`. The
// table codes the `wedge_index ∈ 0..WEDGE_TYPES` element read by two
// §5.11 syntax call sites:
//   * §5.11.28 `read_interintra_mode` — the inter-intra wedge branch,
//     reached when `wedge_interintra == 1`.
//   * §5.11.29 `read_compound_type` — the inter-inter `COMPOUND_WEDGE`
//     branch, reached when `compound_type == COMPOUND_WEDGE` (gated on
//     `Wedge_Bits[ MiSize ] > 0`).
// Both call sites read the same default CDF; the §8.3.2 selection text
// is "wedge_index: The cdf is given by TileWedgeIndexCdf[ MiSize ]".
//
// Per the §9.4 note (p.436), only first-dimension indices 3..=9 (the
// `BLOCK_8X8`..`BLOCK_32X32` band) and 18..=19 (`BLOCK_8X16` /
// `BLOCK_16X8` 2:1 rectangles) are reachable — matching the rows where
// the §3 `Wedge_Bits[ BLOCK_SIZES ]` table is non-zero. The remaining
// rows hold a placeholder uniform CDF (step 2048 = 32768 / 16) and are
// transcribed full-width to keep the indexing uniform with every other
// `[ BLOCK_SIZES ]` table.
// ---------------------------------------------------------------------

/// `Default_Wedge_Index_Cdf[ BLOCK_SIZES ][ WEDGE_TYPES + 1 ]` (§9.4,
/// p.435). Symbol `wedge_index` (§5.11.28 wedge sub-branch and §5.11.29
/// `COMPOUND_WEDGE` branch; selects the direction and offset of the
/// wedge mask used during blending). The §8.3.2 selection is a straight
/// `TileWedgeIndexCdf[ MiSize ]` index. Per the §9.4 note only indices
/// `3..=9` and `18..=19` (the rows where `Wedge_Bits[ MiSize ] > 0`) are
/// ever used in the first dimension; the rows outside that set are the
/// placeholder uniform CDF `{ 2048, 4096, …, 30720, 32768, 0 }` and are
/// transcribed full-width to keep `MiSize`-indexing uniform with every
/// other `[ BLOCK_SIZES ]` table. Each row carries `WEDGE_TYPES = 16`
/// cumulative frequencies plus the §8.3 adaptation counter (starts at
/// 0).
pub const DEFAULT_WEDGE_INDEX_CDF: [[u16; WEDGE_TYPES + 1]; BLOCK_SIZES] = [
    // Rows 0..=2 (BLOCK_4X4 / BLOCK_4X8 / BLOCK_8X4) — placeholder
    // uniform, never reached (Wedge_Bits = 0).
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    // Rows 3..=9 (BLOCK_8X8..BLOCK_32X32) — reachable.
    [
        2438, 4440, 6599, 8663, 11005, 12874, 15751, 18094, 20359, 22362, 24127, 25702, 27752,
        29450, 31171, 32768, 0,
    ],
    [
        806, 3266, 6005, 6738, 7218, 7367, 7771, 14588, 16323, 17367, 18452, 19422, 22839, 26127,
        29629, 32768, 0,
    ],
    [
        2779, 3738, 4683, 7213, 7775, 8017, 8655, 14357, 17939, 21332, 24520, 27470, 29456, 30529,
        31656, 32768, 0,
    ],
    [
        1684, 3625, 5675, 7108, 9302, 11274, 14429, 17144, 19163, 20961, 22884, 24471, 26719,
        28714, 30877, 32768, 0,
    ],
    [
        1142, 3491, 6277, 7314, 8089, 8355, 9023, 13624, 15369, 16730, 18114, 19313, 22521, 26012,
        29550, 32768, 0,
    ],
    [
        2742, 4195, 5727, 8035, 8980, 9336, 10146, 14124, 17270, 20533, 23434, 25972, 27944, 29570,
        31416, 32768, 0,
    ],
    [
        1727, 3948, 6101, 7796, 9841, 12344, 15766, 18944, 20638, 22038, 23963, 25311, 26988,
        28766, 31012, 32768, 0,
    ],
    // Rows 10..=17 — placeholder uniform.
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    // Rows 18..=19 (BLOCK_8X16 / BLOCK_16X8) — reachable.
    [
        154, 987, 1925, 2051, 2088, 2111, 2151, 23033, 23703, 24284, 24985, 25684, 27259, 28883,
        30911, 32768, 0,
    ],
    [
        1135, 1322, 1493, 2635, 2696, 2737, 2770, 21016, 22935, 25057, 27251, 29173, 30089, 30960,
        31933, 32768, 0,
    ],
    // Rows 20..=21 — placeholder uniform.
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
    [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32768, 0,
    ],
];

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

    // -----------------------------------------------------------------
    // Round 137 — intra-frame transform-type group tests.
    // -----------------------------------------------------------------

    /// §9.4 default tables: every row terminates with `1 << 15` and a
    /// zero adaptation counter (§8.2.6 contract). Locks each
    /// transcribed intra-tx-type default and confirms its §3 dimensions.
    #[test]
    fn intra_tx_type_default_tables_well_formed() {
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for plane in &DEFAULT_INTRA_TX_TYPE_SET1_CDF {
            for row in plane {
                check(row);
            }
        }
        for plane in &DEFAULT_INTRA_TX_TYPE_SET2_CDF {
            for row in plane {
                check(row);
            }
        }

        // §3 dimensions.
        assert_eq!(
            DEFAULT_INTRA_TX_TYPE_SET1_CDF.len(),
            INTRA_TX_TYPE_SET1_SIZES
        );
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[0].len(), INTRA_MODES);
        assert_eq!(
            DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][0].len(),
            TX_TYPES_INTRA_SET1 + 1
        );

        assert_eq!(
            DEFAULT_INTRA_TX_TYPE_SET2_CDF.len(),
            INTRA_TX_TYPE_SET2_SIZES
        );
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET2_CDF[0].len(), INTRA_MODES);
        assert_eq!(
            DEFAULT_INTRA_TX_TYPE_SET2_CDF[0][0].len(),
            TX_TYPES_INTRA_SET2 + 1
        );
    }

    /// Spot-check the first / last cumulative frequency in each §9.4
    /// intra-tx-type transcribed table. A mis-keyed digit during
    /// transcription breaks the equality.
    #[test]
    fn intra_tx_type_table_byte_anchors() {
        // Default_Intra_Tx_Type_Set1_Cdf — TX_4X4 (size 0).
        // Row 0 (DC_PRED): { 1535, 8035, 9461, 12751, 23467, 27825, 32768, 0 }
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][0][0], 1535);
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][0][5], 27825);
        // Row 12 (PAETH_PRED, last): { 3409, 5436, 10599, 15599, 19687, 24040, 32768, 0 }
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][12][0], 3409);
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][12][5], 24040);

        // Default_Intra_Tx_Type_Set1_Cdf — TX_8X8 (size 1).
        // Row 0 (DC_PRED): { 1870, 13742, 14530, 16498, 23770, 27698, 32768, 0 }
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[1][0][0], 1870);
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[1][0][5], 27698);
        // Row 12 (PAETH_PRED): { 3511, 6332, 11165, 15335, 19323, 23594, 32768, 0 }
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[1][12][0], 3511);
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[1][12][5], 23594);

        // Default_Intra_Tx_Type_Set2_Cdf — TX_4X4 (size 0) and TX_8X8
        // (size 1) are the §9.4 flat 5-way distribution.
        for (size_idx, plane) in DEFAULT_INTRA_TX_TYPE_SET2_CDF.iter().enumerate().take(2) {
            for (y, row) in plane.iter().enumerate() {
                assert_eq!(
                    row,
                    &[6554, 13107, 19661, 26214, 32768, 0],
                    "Set2 size {size_idx} row {y} must be the §9.4 flat distribution"
                );
            }
        }
        // Default_Intra_Tx_Type_Set2_Cdf — TX_16X16 (size 2).
        // Row 0 (DC_PRED): { 1127, 12814, 22772, 27483, 32768, 0 }
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET2_CDF[2][0][0], 1127);
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET2_CDF[2][0][3], 27483);
        // Row 12 (PAETH_PRED): { 1968, 5556, 12023, 18547, 32768, 0 }
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET2_CDF[2][12][0], 1968);
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET2_CDF[2][12][3], 18547);
    }

    /// §8.3.1: a fresh context copies every intra-tx-type default in
    /// (independent of the source — the §9.4 array is not aliased).
    #[test]
    fn intra_tx_type_init_from_defaults_copies_tables() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.intra_tx_type_set1, DEFAULT_INTRA_TX_TYPE_SET1_CDF);
        assert_eq!(ctx.intra_tx_type_set2, DEFAULT_INTRA_TX_TYPE_SET2_CDF);

        // Working-copy independence: mutating the context must not
        // touch the §9.4 source.
        ctx.intra_tx_type_cdf(TX_SET_INTRA_1, 0, 0).unwrap()[0] = 12345;
        assert_ne!(
            ctx.intra_tx_type_set1[0][0],
            DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][0]
        );
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][0][0], 1535);
    }

    /// §8.3.2 `intra_tx_type` selection: each `(set, tx_size_sqr,
    /// intra_dir)` triple picks the expected
    /// `Default_Intra_Tx_Type_Set*_Cdf` row, the row length matches the
    /// spec's per-set symbol count, and the unreachable /
    /// `TX_SET_DCTONLY` paths return `None`.
    #[test]
    fn intra_tx_type_cdf_selected_by_set() {
        let mut ctx = TileCdfContext::new_from_defaults();

        // TX_SET_INTRA_1, Tx_Size_Sqr = TX_4X4 (=0), intraDir = DC_PRED (=0).
        let row1 = ctx.intra_tx_type_cdf(TX_SET_INTRA_1, 0, 0).unwrap();
        assert_eq!(row1.len(), TX_TYPES_INTRA_SET1 + 1);
        assert_eq!(row1, &DEFAULT_INTRA_TX_TYPE_SET1_CDF[0][0]);
        // TX_SET_INTRA_1, Tx_Size_Sqr = TX_8X8 (=1), intraDir = PAETH_PRED (=12).
        let row1b = ctx.intra_tx_type_cdf(TX_SET_INTRA_1, 1, 12).unwrap();
        assert_eq!(row1b, &DEFAULT_INTRA_TX_TYPE_SET1_CDF[1][12]);

        // TX_SET_INTRA_2 across its three reachable sizes.
        for sqr in 0..INTRA_TX_TYPE_SET2_SIZES as u32 {
            for (dir, expected_row) in DEFAULT_INTRA_TX_TYPE_SET2_CDF[sqr as usize]
                .iter()
                .enumerate()
            {
                let row = ctx.intra_tx_type_cdf(TX_SET_INTRA_2, sqr, dir).unwrap();
                assert_eq!(row.len(), TX_TYPES_INTRA_SET2 + 1);
                assert_eq!(row, expected_row);
            }
        }

        // TX_SET_DCTONLY: §5.11.47 forces TxType = DCT_DCT and skips
        // the intra_tx_type read; the selector returns None.
        assert!(ctx.intra_tx_type_cdf(TX_SET_DCTONLY, 0, 0).is_none());

        // Out-of-range tx_size_sqr for Set1 (only TX_4X4 / TX_8X8) /
        // Set2 (only TX_4X4 / TX_8X8 / TX_16X16).
        assert!(ctx
            .intra_tx_type_cdf(TX_SET_INTRA_1, INTRA_TX_TYPE_SET1_SIZES as u32, 0)
            .is_none());
        assert!(ctx
            .intra_tx_type_cdf(TX_SET_INTRA_2, INTRA_TX_TYPE_SET2_SIZES as u32, 0)
            .is_none());

        // Out-of-range intra_dir.
        assert!(ctx
            .intra_tx_type_cdf(TX_SET_INTRA_1, 0, INTRA_MODES)
            .is_none());
        assert!(ctx
            .intra_tx_type_cdf(TX_SET_INTRA_2, 2, INTRA_MODES)
            .is_none());

        // Out-of-range set (the inter set IDs share the same numeric
        // space — but the intra selector should not return a row for an
        // unreachable set ⇒ Set3 isn't an intra set; the inter values
        // are themselves intra `1` / `2`, so a stray `set == 3` is
        // invalid for the intra branch).
        assert!(ctx.intra_tx_type_cdf(99, 0, 0).is_none());
    }

    /// §5.11.48 `get_tx_set()` (intra branch). Walk every reachable
    /// `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)` and confirm the
    /// helper returns the spec-prescribed set.
    #[test]
    fn intra_tx_type_set_get_tx_set_intra_branch() {
        // TX_4X4 / TX_8X8 with txSzSqrUp == txSzSqr, !reduced ⇒
        // TX_SET_INTRA_1.
        assert_eq!(intra_tx_type_set(0, 0, false), TX_SET_INTRA_1);
        assert_eq!(intra_tx_type_set(1, 1, false), TX_SET_INTRA_1);

        // TX_16X16 (txSzSqr == 2), !reduced ⇒ TX_SET_INTRA_2.
        assert_eq!(intra_tx_type_set(2, 2, false), TX_SET_INTRA_2);

        // TX_32X32 (txSzSqrUp == 3), !reduced ⇒ TX_SET_DCTONLY (the
        // §5.11.48 intra branch is stricter than the inter one — any
        // `txSzSqrUp == TX_32X32` returns `TX_SET_DCTONLY`).
        assert_eq!(intra_tx_type_set(3, 3, false), TX_SET_DCTONLY);

        // Reduced-tx-set forces TX_SET_INTRA_2 for sqrUp < 32x32.
        assert_eq!(intra_tx_type_set(0, 0, true), TX_SET_INTRA_2);
        assert_eq!(intra_tx_type_set(1, 1, true), TX_SET_INTRA_2);
        assert_eq!(intra_tx_type_set(2, 2, true), TX_SET_INTRA_2);

        // txSzSqrUp >= TX_32X32 ⇒ TX_SET_DCTONLY regardless of
        // reduced_tx_set / txSzSqr.
        assert_eq!(intra_tx_type_set(3, 3, true), TX_SET_DCTONLY);
        assert_eq!(intra_tx_type_set(0, 4, false), TX_SET_DCTONLY);
        assert_eq!(intra_tx_type_set(3, 4, true), TX_SET_DCTONLY);

        // The rectangular-tx case: a TX_4X8 block has txSzSqr = TX_4X4
        // (0), txSzSqrUp = TX_8X8 (1). No reduced_tx_set ⇒
        // TX_SET_INTRA_1.
        assert_eq!(intra_tx_type_set(0, 1, false), TX_SET_INTRA_1);
        // A TX_16X32 / TX_32X16: txSzSqr = TX_16X16 (2),
        // txSzSqrUp = TX_32X32 (3) ⇒ TX_SET_DCTONLY (the `txSzSqrUp ==
        // TX_32X32` gate fires before the `txSzSqr == TX_16X16` branch
        // on the intra branch).
        assert_eq!(intra_tx_type_set(2, 3, false), TX_SET_DCTONLY);
    }

    /// §8.3.2 `intraDir` derivation: with `use_filter_intra == 0` the
    /// caller's `YMode` is passed through; with `use_filter_intra == 1`
    /// the `Filter_Intra_Mode_To_Intra_Dir` table maps each filter mode
    /// to a directional anchor (DC_PRED / V_PRED / H_PRED / D157_PRED /
    /// DC_PRED).
    #[test]
    fn intra_dir_derivation() {
        // Non-filter-intra: pass-through YMode.
        for y in 0..INTRA_MODES {
            assert_eq!(intra_dir(false, y, 0), Some(y));
        }
        // Out-of-range YMode (pass-through bound) returns None.
        assert!(intra_dir(false, INTRA_MODES, 0).is_none());

        // Filter-intra: Filter_Intra_Mode_To_Intra_Dir lookup.
        // The spec listing reads { DC_PRED, V_PRED, H_PRED, D157_PRED,
        // DC_PRED }, i.e. { 0, 1, 2, 6, 0 }.
        assert_eq!(intra_dir(true, 99, 0), Some(0)); // FILTER_DC_PRED → DC_PRED
        assert_eq!(intra_dir(true, 99, 1), Some(1)); // FILTER_V_PRED → V_PRED
        assert_eq!(intra_dir(true, 99, 2), Some(2)); // FILTER_H_PRED → H_PRED
        assert_eq!(intra_dir(true, 99, 3), Some(6)); // FILTER_D157_PRED → D157_PRED
        assert_eq!(intra_dir(true, 99, 4), Some(0)); // FILTER_PAETH_PRED → DC_PRED

        // Out-of-range filter_intra_mode returns None.
        assert!(intra_dir(true, 0, INTRA_FILTER_MODES).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through an
    /// `intra_tx_type` default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the decode
    /// lands in range and the working copy adapts.
    #[test]
    fn decode_intra_tx_type_through_default_cdf() {
        // Pick TX_SET_INTRA_2 (5-symbol row), TX_16X16 (Tx_Size_Sqr =
        // 2). intraDir = DC_PRED (the §9.4 row { 1127, 12814, 22772,
        // 27483, 32768, 0 }).
        let set = intra_tx_type_set(2, 2, false);
        assert_eq!(set, TX_SET_INTRA_2);
        let dir = intra_dir(false, 0, 0).unwrap();
        assert_eq!(dir, 0);

        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.intra_tx_type_set2;

        let row = ctx.intra_tx_type_cdf(set, 2, dir).unwrap();
        assert_eq!(row, &DEFAULT_INTRA_TX_TYPE_SET2_CDF[2][0]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < TX_TYPES_INTRA_SET2,
            "intra_tx_type via Set2 codes one of \
             {{ IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST }}"
        );

        assert_ne!(
            ctx.intra_tx_type_set2, before,
            "read_symbol must adapt the CDF"
        );
        assert_eq!(DEFAULT_INTRA_TX_TYPE_SET2_CDF[2][0][0], 1127);
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

    // Round 23 — motion-mode group tests.

    /// §8.3.1 / §9.4: the `Default_Motion_Mode_Cdf` table is
    /// well-formed. Outer dim matches `BLOCK_SIZES`; each row is
    /// `[MOTION_MODES + 1]` (3 cumulative frequencies + adaptation
    /// counter); the second-to-last entry is `1 << 15` and the last is
    /// 0.
    #[test]
    fn motion_mode_default_table_well_formed() {
        assert_eq!(DEFAULT_MOTION_MODE_CDF.len(), BLOCK_SIZES);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[0].len(), MOTION_MODES + 1);
        // §3 constant pinned: SIMPLE/OBMC/LOCALWARP ⇒ 3 motion modes.
        assert_eq!(MOTION_MODES, 3);
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for r in &DEFAULT_MOTION_MODE_CDF {
            check(r);
        }
    }

    /// Spot-check the §9.4 `Default_Motion_Mode_Cdf` values byte-for-byte.
    /// A mis-keyed digit during transcription breaks the equality.
    #[test]
    fn motion_mode_default_byte_exact_values() {
        // Rows 0..=2: the spec-flagged-unreachable rows initialise to
        // the flat `{ 10923, 21845, 32768, 0 }` — three roughly equal
        // buckets (≈ 10923 = 32768/3).
        assert_eq!(DEFAULT_MOTION_MODE_CDF[0], [10923, 21845, 32768, 0]);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[1], [10923, 21845, 32768, 0]);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[2], [10923, 21845, 32768, 0]);
        // Row 3: { 7651, 24760, 32768, 0 } — first reachable row.
        assert_eq!(DEFAULT_MOTION_MODE_CDF[3][0], 7651);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[3][1], 24760);
        // Row 9: { 26260, 29116, 32768, 0 } — strong-SIMPLE bias anchor.
        assert_eq!(DEFAULT_MOTION_MODE_CDF[9][0], 26260);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[9][1], 29116);
        // Row 15: { 32507, 32558, 32768, 0 } — heaviest SIMPLE bias.
        assert_eq!(DEFAULT_MOTION_MODE_CDF[15][0], 32507);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[15][1], 32558);
        // Rows 16..=17: spec-unreachable, flat-init.
        assert_eq!(DEFAULT_MOTION_MODE_CDF[16], [10923, 21845, 32768, 0]);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[17], [10923, 21845, 32768, 0]);
        // Row 21: { 29742, 31203, 32768, 0 } — last row anchor.
        assert_eq!(DEFAULT_MOTION_MODE_CDF[21][0], 29742);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[21][1], 31203);
    }

    /// §8.3.1: a fresh context copies the motion-mode default in (the
    /// §9.4 source is not aliased).
    #[test]
    fn motion_mode_init_from_defaults_copies_table() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.motion_mode, DEFAULT_MOTION_MODE_CDF);

        // Working-copy independence: mutating the context must not
        // touch the §9.4 source.
        ctx.motion_mode_cdf(3).unwrap()[0] = 12345;
        assert_ne!(ctx.motion_mode[3][0], DEFAULT_MOTION_MODE_CDF[3][0]);
        assert_eq!(DEFAULT_MOTION_MODE_CDF[3][0], 7651);
    }

    /// §8.3.2 `motion_mode` selector returns the right §9.4 row for
    /// every `MiSize`, with row length matching the spec
    /// (`MOTION_MODES + 1`).
    #[test]
    fn motion_mode_selector_returns_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();
        for (i, want) in DEFAULT_MOTION_MODE_CDF.iter().enumerate() {
            let row = ctx.motion_mode_cdf(i).unwrap();
            assert_eq!(row.len(), MOTION_MODES + 1);
            assert_eq!(row, want);
        }
        // Out-of-range `MiSize` returns None.
        assert!(ctx.motion_mode_cdf(BLOCK_SIZES).is_none());
        assert!(ctx.motion_mode_cdf(BLOCK_SIZES + 7).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// `motion_mode` default CDF selected by the §8.3.2 selection
    /// (`MiSize = 15`, the heaviest-SIMPLE-bias row), confirming the
    /// chosen row matches the §9.4 source, the decode lands in range
    /// and the working copy adapts.
    #[test]
    fn decode_motion_mode_through_default_cdf() {
        // Default_Motion_Mode_Cdf[15] = { 32507, 32558, 32768, 0 } —
        // very strong SIMPLE bias. Drive the decoder with a typical
        // start-of-tile window and assert the working copy mutates and
        // the §9.4 source stays put.
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.motion_mode;

        let row = tile_ctx.motion_mode_cdf(15).unwrap();
        assert_eq!(row, &DEFAULT_MOTION_MODE_CDF[15]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < MOTION_MODES,
            "motion_mode must code a symbol in 0..MOTION_MODES"
        );

        assert_ne!(
            tile_ctx.motion_mode, before,
            "read_symbol must adapt the working CDF"
        );
        assert_eq!(DEFAULT_MOTION_MODE_CDF[15], [32507, 32558, 32768, 0]);
    }

    // Round 24 — compound-prediction group tests.

    /// §8.3.1 / §9.4: the three compound-prediction default tables are
    /// well-formed. Outer dims match the §3 constants; each binary row
    /// is `[3]` (`comp_group_idx` / `compound_idx`) or
    /// `[COMPOUND_TYPES + 1]` (`compound_type`); the second-to-last
    /// entry is `1 << 15` and the last is a fresh-0 adaptation counter.
    #[test]
    fn compound_default_tables_well_formed() {
        // §3 constants pinned.
        assert_eq!(COMPOUND_TYPES, 2);
        assert_eq!(COMP_GROUP_IDX_CONTEXTS, 6);
        assert_eq!(COMPOUND_IDX_CONTEXTS, 6);

        assert_eq!(DEFAULT_COMP_GROUP_IDX_CDF.len(), COMP_GROUP_IDX_CONTEXTS);
        assert_eq!(DEFAULT_COMPOUND_IDX_CDF.len(), COMPOUND_IDX_CONTEXTS);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF.len(), BLOCK_SIZES);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[0].len(), COMPOUND_TYPES + 1);

        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for r in &DEFAULT_COMP_GROUP_IDX_CDF {
            assert_eq!(r.len(), 3);
            check(r);
        }
        for r in &DEFAULT_COMPOUND_IDX_CDF {
            assert_eq!(r.len(), 3);
            check(r);
        }
        for r in &DEFAULT_COMPOUND_TYPE_CDF {
            check(r);
        }
    }

    /// Spot-check the §9.4 compound default values byte-for-byte. A
    /// mis-keyed digit during transcription breaks the equality.
    #[test]
    fn compound_default_byte_exact_values() {
        // Default_Compound_Idx_Cdf — 6 binary context rows.
        assert_eq!(DEFAULT_COMPOUND_IDX_CDF[0], [18244, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_IDX_CDF[2], [7053, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_IDX_CDF[5], [4644, 32768, 0]);

        // Default_Comp_Group_Idx_Cdf — 6 binary context rows.
        assert_eq!(DEFAULT_COMP_GROUP_IDX_CDF[0], [26607, 32768, 0]);
        assert_eq!(DEFAULT_COMP_GROUP_IDX_CDF[2], [18840, 32768, 0]);
        assert_eq!(DEFAULT_COMP_GROUP_IDX_CDF[5], [22674, 32768, 0]);

        // Default_Compound_Type_Cdf — the §9.4 note marks rows 0..=2,
        // 10..=17 and 20..=21 unreachable; those carry the flat
        // { 16384, 32768, 0 } placeholder.
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[0], [16384, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[2], [16384, 32768, 0]);
        // Row 3: { 23431, 32768, 0 } — first reachable row.
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[3], [23431, 32768, 0]);
        // Row 9: { 6172, 32768, 0 } — last of the 3..=9 reachable run.
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[9], [6172, 32768, 0]);
        // Rows 10..=17: spec-unreachable, flat-init.
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[10], [16384, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[17], [16384, 32768, 0]);
        // Rows 18 / 19: { 11820, .. } / { 7701, .. } — second reachable run.
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[18], [11820, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[19], [7701, 32768, 0]);
        // Rows 20..=21: spec-unreachable, flat-init.
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[20], [16384, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[21], [16384, 32768, 0]);
    }

    /// §8.3.1: a fresh context copies the three compound-prediction
    /// defaults in (the §9.4 sources are not aliased).
    #[test]
    fn compound_init_from_defaults_copies_tables() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.comp_group_idx, DEFAULT_COMP_GROUP_IDX_CDF);
        assert_eq!(ctx.compound_idx, DEFAULT_COMPOUND_IDX_CDF);
        assert_eq!(ctx.compound_type, DEFAULT_COMPOUND_TYPE_CDF);

        // Working-copy independence: mutating the context must not touch
        // the §9.4 sources.
        ctx.comp_group_idx_cdf(0)[0] = 12345;
        ctx.compound_idx_cdf(0)[0] = 23456;
        ctx.compound_type_cdf(3).unwrap()[0] = 34567;
        assert_ne!(ctx.comp_group_idx[0][0], DEFAULT_COMP_GROUP_IDX_CDF[0][0]);
        assert_ne!(ctx.compound_idx[0][0], DEFAULT_COMPOUND_IDX_CDF[0][0]);
        assert_ne!(ctx.compound_type[3][0], DEFAULT_COMPOUND_TYPE_CDF[3][0]);
        assert_eq!(DEFAULT_COMP_GROUP_IDX_CDF[0][0], 26607);
        assert_eq!(DEFAULT_COMPOUND_IDX_CDF[0][0], 18244);
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[3][0], 23431);
    }

    /// §8.3.2 compound selectors return the right §9.4 rows. The two
    /// binary selectors take a precomputed `ctx`; the `compound_type`
    /// selector takes `MiSize` and returns `None` out of range.
    #[test]
    fn compound_selectors_return_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();
        for (i, want) in DEFAULT_COMP_GROUP_IDX_CDF.iter().enumerate() {
            let row = ctx.comp_group_idx_cdf(i);
            assert_eq!(row.len(), 3);
            assert_eq!(row, want);
        }
        for (i, want) in DEFAULT_COMPOUND_IDX_CDF.iter().enumerate() {
            let row = ctx.compound_idx_cdf(i);
            assert_eq!(row.len(), 3);
            assert_eq!(row, want);
        }
        for (i, want) in DEFAULT_COMPOUND_TYPE_CDF.iter().enumerate() {
            let row = ctx.compound_type_cdf(i).unwrap();
            assert_eq!(row.len(), COMPOUND_TYPES + 1);
            assert_eq!(row, want);
        }
        // Out-of-range `MiSize` returns None.
        assert!(ctx.compound_type_cdf(BLOCK_SIZES).is_none());
        assert!(ctx.compound_type_cdf(BLOCK_SIZES + 7).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// `compound_type` default CDF selected by the §8.3.2 selection
    /// (`MiSize = 9`, the strongest-bias reachable row), confirming the
    /// chosen row matches the §9.4 source, the decode lands in range and
    /// the working copy adapts.
    #[test]
    fn decode_compound_type_through_default_cdf() {
        // Default_Compound_Type_Cdf[9] = { 6172, 32768, 0 }.
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.compound_type;

        let row = tile_ctx.compound_type_cdf(9).unwrap();
        assert_eq!(row, &DEFAULT_COMPOUND_TYPE_CDF[9]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < COMPOUND_TYPES,
            "compound_type must code a symbol in 0..COMPOUND_TYPES"
        );

        assert_ne!(
            tile_ctx.compound_type, before,
            "read_symbol must adapt the working CDF"
        );
        assert_eq!(DEFAULT_COMPOUND_TYPE_CDF[9], [6172, 32768, 0]);
    }

    /// End-to-end: drive the §8.2 `SymbolDecoder` through a binary
    /// `comp_group_idx` default CDF (ctx = 2, the lowest-bias context),
    /// confirming the row matches §9.4, the decode is in range and the
    /// working copy adapts.
    #[test]
    fn decode_comp_group_idx_through_default_cdf() {
        // Default_Comp_Group_Idx_Cdf[2] = { 18840, 32768, 0 }.
        let bytes = [0x40u8, 0x00u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.comp_group_idx;

        let row = tile_ctx.comp_group_idx_cdf(2);
        assert_eq!(row, &DEFAULT_COMP_GROUP_IDX_CDF[2]);
        let sym = dec.read_symbol(row).unwrap();
        assert!((sym as usize) < 2, "comp_group_idx is a binary symbol");

        assert_ne!(
            tile_ctx.comp_group_idx, before,
            "read_symbol must adapt the working CDF"
        );
        assert_eq!(DEFAULT_COMP_GROUP_IDX_CDF[2], [18840, 32768, 0]);
    }

    // Round 134 — inter-frame intra-mode group tests.

    /// §8.3.1 / §9.4: the three inter-frame intra-mode default tables are
    /// well-formed. Outer dims match the §3 constants; each row carries
    /// the right number of cumulative frequencies + the adaptation
    /// counter; the second-to-last entry is `1 << 15` and the last is a
    /// fresh-0 adaptation counter; the cumulative frequencies are
    /// strictly increasing up to `32768`.
    #[test]
    fn inter_intra_mode_default_tables_well_formed() {
        // §3 constants pinned.
        assert_eq!(BLOCK_SIZE_GROUPS, 4);
        assert_eq!(UV_INTRA_MODES_CFL_NOT_ALLOWED, 13);
        assert_eq!(UV_INTRA_MODES_CFL_ALLOWED, 14);
        assert_eq!(INTRA_MODES, 13);

        // Outer dimensions.
        assert_eq!(DEFAULT_Y_MODE_CDF.len(), BLOCK_SIZE_GROUPS);
        assert_eq!(DEFAULT_Y_MODE_CDF[0].len(), INTRA_MODES + 1);
        assert_eq!(DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF.len(), INTRA_MODES);
        assert_eq!(
            DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[0].len(),
            UV_INTRA_MODES_CFL_NOT_ALLOWED + 1
        );
        assert_eq!(DEFAULT_UV_MODE_CFL_ALLOWED_CDF.len(), INTRA_MODES);
        assert_eq!(
            DEFAULT_UV_MODE_CFL_ALLOWED_CDF[0].len(),
            UV_INTRA_MODES_CFL_ALLOWED + 1
        );

        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
            // Cumulative frequencies strictly increase to 32768.
            for w in row[..n].windows(2) {
                assert!(w[0] < w[1], "cdf must be strictly increasing: {row:?}");
            }
        };
        for r in &DEFAULT_Y_MODE_CDF {
            check(r);
        }
        for r in &DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF {
            check(r);
        }
        for r in &DEFAULT_UV_MODE_CFL_ALLOWED_CDF {
            check(r);
        }
    }

    /// Spot-check the §9.4 inter-frame intra-mode default values
    /// byte-for-byte. A mis-keyed digit during transcription breaks the
    /// equality.
    #[test]
    fn inter_intra_mode_default_byte_exact_values() {
        // Default_Y_Mode_Cdf — first / last context rows.
        assert_eq!(
            DEFAULT_Y_MODE_CDF[0],
            [
                22801, 23489, 24293, 24756, 25601, 26123, 26606, 27418, 27945, 29228, 29685, 30349,
                32768, 0
            ]
        );
        assert_eq!(
            DEFAULT_Y_MODE_CDF[3],
            [
                20155, 21301, 22838, 23178, 23261, 23533, 23703, 24804, 25352, 26575, 27016, 28049,
                32768, 0
            ]
        );
        // Leading-frequency anchors across the four contexts.
        assert_eq!(DEFAULT_Y_MODE_CDF[1][0], 18673);
        assert_eq!(DEFAULT_Y_MODE_CDF[2][0], 19770);

        // Default_Uv_Mode_Cfl_Not_Allowed_Cdf — first / last YMode rows.
        assert_eq!(
            DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[0],
            [
                22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923, 28244, 30059, 30941, 31961,
                32768, 0
            ]
        );
        assert_eq!(
            DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[12],
            [
                12124, 13759, 14959, 14992, 15007, 15051, 15078, 15166, 15255, 15753, 16039, 16606,
                32768, 0
            ]
        );
        assert_eq!(DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[1][0], 9513);
        assert_eq!(DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[6][6], 28551);

        // Default_Uv_Mode_Cfl_Allowed_Cdf — first / last YMode rows.
        assert_eq!(
            DEFAULT_UV_MODE_CFL_ALLOWED_CDF[0],
            [
                10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656, 15986, 20086, 20995, 22455,
                24212, 32768, 0
            ]
        );
        assert_eq!(
            DEFAULT_UV_MODE_CFL_ALLOWED_CDF[12],
            [
                3144, 5087, 7382, 7504, 7593, 7690, 7801, 8064, 8232, 9248, 9875, 10521, 29048,
                32768, 0
            ]
        );
        assert_eq!(DEFAULT_UV_MODE_CFL_ALLOWED_CDF[1][0], 4532);
        assert_eq!(DEFAULT_UV_MODE_CFL_ALLOWED_CDF[8][8], 18898);
    }

    /// §8.3.2 `Size_Group[ BLOCK_SIZES ]` table + [`size_group`] helper:
    /// pinned byte-for-byte and confirmed to map into `0..BLOCK_SIZE_GROUPS`.
    #[test]
    fn size_group_table_pinned() {
        assert_eq!(SIZE_GROUP.len(), BLOCK_SIZES);
        assert_eq!(
            SIZE_GROUP,
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2]
        );
        for (mi_size, &want) in SIZE_GROUP.iter().enumerate() {
            let g = size_group(mi_size);
            assert!(g < BLOCK_SIZE_GROUPS, "Size_Group must index a y_mode ctx");
            assert_eq!(g, want);
        }
    }

    /// §8.3.1: a fresh context copies the three inter-frame intra-mode
    /// defaults in (the §9.4 sources are not aliased).
    #[test]
    fn inter_intra_mode_init_from_defaults_copies_tables() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.y_mode, DEFAULT_Y_MODE_CDF);
        assert_eq!(
            ctx.uv_mode_cfl_not_allowed,
            DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF
        );
        assert_eq!(ctx.uv_mode_cfl_allowed, DEFAULT_UV_MODE_CFL_ALLOWED_CDF);

        // Working-copy independence: mutating the context must not touch
        // the §9.4 sources.
        ctx.y_mode_cdf(0).unwrap()[0] = 12345;
        ctx.uv_mode_cdf(false, 0).unwrap()[0] = 23456;
        ctx.uv_mode_cdf(true, 0).unwrap()[0] = 34567;
        assert_ne!(ctx.y_mode[0][0], DEFAULT_Y_MODE_CDF[0][0]);
        assert_ne!(
            ctx.uv_mode_cfl_not_allowed[0][0],
            DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[0][0]
        );
        assert_ne!(
            ctx.uv_mode_cfl_allowed[0][0],
            DEFAULT_UV_MODE_CFL_ALLOWED_CDF[0][0]
        );
        assert_eq!(DEFAULT_Y_MODE_CDF[0][0], 22801);
        assert_eq!(DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[0][0], 22631);
        assert_eq!(DEFAULT_UV_MODE_CFL_ALLOWED_CDF[0][0], 10407);
    }

    /// §8.3.2 inter-frame intra-mode selectors return the right §9.4 rows.
    /// `y_mode_cdf` takes `ctx = Size_Group[MiSize]`; `uv_mode_cdf` picks
    /// the cfl-allowed / cfl-not-allowed variant by the resolved flag and
    /// indexes by `YMode`. Both reject out-of-range indices with `None`.
    #[test]
    fn inter_intra_mode_selectors_return_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();

        for (i, want) in DEFAULT_Y_MODE_CDF.iter().enumerate() {
            let row = ctx.y_mode_cdf(i).unwrap();
            assert_eq!(row.len(), INTRA_MODES + 1);
            assert_eq!(row, want);
        }
        assert!(ctx.y_mode_cdf(BLOCK_SIZE_GROUPS).is_none());
        assert!(ctx.y_mode_cdf(BLOCK_SIZE_GROUPS + 3).is_none());

        for (i, want) in DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF.iter().enumerate() {
            let row = ctx.uv_mode_cdf(false, i).unwrap();
            assert_eq!(row.len(), UV_INTRA_MODES_CFL_NOT_ALLOWED + 1);
            assert_eq!(row, want);
        }
        for (i, want) in DEFAULT_UV_MODE_CFL_ALLOWED_CDF.iter().enumerate() {
            let row = ctx.uv_mode_cdf(true, i).unwrap();
            assert_eq!(row.len(), UV_INTRA_MODES_CFL_ALLOWED + 1);
            assert_eq!(row, want);
        }
        // Out-of-range `YMode` returns None for both variants.
        assert!(ctx.uv_mode_cdf(false, INTRA_MODES).is_none());
        assert!(ctx.uv_mode_cdf(true, INTRA_MODES + 5).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a `y_mode`
    /// default CDF selected by the §8.3.2 selection
    /// (`ctx = Size_Group[ MiSize ]` for a 64×64 `MiSize`, the
    /// largest-block context 3), confirming the chosen row matches the
    /// §9.4 source, the decode lands in range and the working copy adapts.
    #[test]
    fn decode_y_mode_through_default_cdf() {
        // MiSize = 12 (BLOCK_64X64) → Size_Group[12] = 3.
        assert_eq!(size_group(12), 3);
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.y_mode;

        let ctx = size_group(12);
        let row = tile_ctx.y_mode_cdf(ctx).unwrap();
        assert_eq!(row, &DEFAULT_Y_MODE_CDF[3]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < INTRA_MODES,
            "y_mode must code a symbol in 0..INTRA_MODES"
        );

        assert_ne!(
            tile_ctx.y_mode, before,
            "read_symbol must adapt the working CDF"
        );
        assert_eq!(DEFAULT_Y_MODE_CDF[3][0], 20155);
    }

    /// End-to-end: drive the §8.2 `SymbolDecoder` through both `uv_mode`
    /// variants (`cfl_allowed` true / false) for the same `YMode`,
    /// confirming each selector picks the correct §9.4 table, the decode
    /// lands in the matching value range and the working copy adapts.
    #[test]
    fn decode_uv_mode_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];

        // cfl-not-allowed: 13 coded values, YMode = 0.
        {
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut tile_ctx = TileCdfContext::new_from_defaults();
            let before = tile_ctx.uv_mode_cfl_not_allowed;
            let row = tile_ctx.uv_mode_cdf(false, 0).unwrap();
            assert_eq!(row, &DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[0]);
            let sym = dec.read_symbol(row).unwrap();
            assert!(
                (sym as usize) < UV_INTRA_MODES_CFL_NOT_ALLOWED,
                "uv_mode (cfl-not-allowed) must code 0..UV_INTRA_MODES_CFL_NOT_ALLOWED"
            );
            assert_ne!(
                tile_ctx.uv_mode_cfl_not_allowed, before,
                "read_symbol must adapt the working CDF"
            );
        }

        // cfl-allowed: 14 coded values (UV_CFL_PRED included), YMode = 0.
        {
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut tile_ctx = TileCdfContext::new_from_defaults();
            let before = tile_ctx.uv_mode_cfl_allowed;
            let row = tile_ctx.uv_mode_cdf(true, 0).unwrap();
            assert_eq!(row, &DEFAULT_UV_MODE_CFL_ALLOWED_CDF[0]);
            let sym = dec.read_symbol(row).unwrap();
            assert!(
                (sym as usize) < UV_INTRA_MODES_CFL_ALLOWED,
                "uv_mode (cfl-allowed) must code 0..UV_INTRA_MODES_CFL_ALLOWED"
            );
            assert_ne!(
                tile_ctx.uv_mode_cfl_allowed, before,
                "read_symbol must adapt the working CDF"
            );
        }
        assert_eq!(DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF[0][0], 22631);
        assert_eq!(DEFAULT_UV_MODE_CFL_ALLOWED_CDF[0][0], 10407);
    }

    // Round 135 — angle-delta group tests.

    /// §8.3.1 / §9.4: the angle-delta default table is well-formed. The
    /// §3 constants are pinned, the outer dimension matches
    /// `DIRECTIONAL_MODES`, each row carries `2 * MAX_ANGLE_DELTA + 1 = 7`
    /// cumulative frequencies + the adaptation counter, the
    /// second-to-last entry is `1 << 15` and the last is a fresh-0
    /// counter, and the cumulative frequencies strictly increase to
    /// `32768`.
    #[test]
    fn angle_delta_default_table_well_formed() {
        // §3 constants pinned.
        assert_eq!(DIRECTIONAL_MODES, 8);
        assert_eq!(MAX_ANGLE_DELTA, 3);
        assert_eq!(V_PRED, 1);
        assert_eq!(2 * MAX_ANGLE_DELTA + 1, 7);

        // Outer / inner dimensions.
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF.len(), DIRECTIONAL_MODES);
        assert_eq!(
            DEFAULT_ANGLE_DELTA_CDF[0].len(),
            (2 * MAX_ANGLE_DELTA + 1) + 1
        );

        for row in &DEFAULT_ANGLE_DELTA_CDF {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
            for w in row[..n].windows(2) {
                assert!(w[0] < w[1], "cdf must be strictly increasing: {row:?}");
            }
        }
    }

    /// Spot-check the §9.4 angle-delta default values byte-for-byte. A
    /// mis-keyed digit during transcription breaks the equality.
    #[test]
    fn angle_delta_default_byte_exact_values() {
        // First / last directional-mode rows.
        assert_eq!(
            DEFAULT_ANGLE_DELTA_CDF[0],
            [2180, 5032, 7567, 22776, 26989, 30217, 32768, 0]
        );
        assert_eq!(
            DEFAULT_ANGLE_DELTA_CDF[7],
            [3605, 10428, 12459, 17676, 21244, 30655, 32768, 0]
        );
        // A couple of interior rows + leading-frequency anchors.
        assert_eq!(
            DEFAULT_ANGLE_DELTA_CDF[3],
            [4581, 11226, 15147, 17138, 21834, 28397, 32768, 0]
        );
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[1][0], 2301);
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[2][0], 3780);
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[4][0], 1737);
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[5][2], 12485);
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[6][4], 22561);
    }

    /// §8.3.1: a fresh context copies the angle-delta default in (the
    /// §9.4 source is not aliased), and mutating the working copy leaves
    /// the source untouched.
    #[test]
    fn angle_delta_init_from_defaults_copies_table() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.angle_delta, DEFAULT_ANGLE_DELTA_CDF);

        // V_PRED maps to row 0.
        ctx.angle_delta_cdf(V_PRED).unwrap()[0] = 12345;
        assert_ne!(ctx.angle_delta[0][0], DEFAULT_ANGLE_DELTA_CDF[0][0]);
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[0][0], 2180);
    }

    /// §8.3.2 `angle_delta_y` / `angle_delta_uv` selector: takes the
    /// directional `YMode` / `UVMode` and rebases by `V_PRED`, returning
    /// `TileAngleDeltaCdf[ mode - V_PRED ]`. Confirms every directional
    /// mode maps to the right §9.4 row and that non-directional modes
    /// (below `V_PRED`, or at/above `V_PRED + DIRECTIONAL_MODES`) return
    /// `None`.
    #[test]
    fn angle_delta_selector_returns_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();

        // mode = V_PRED .. V_PRED + DIRECTIONAL_MODES - 1 → rows 0..8.
        for (i, want) in DEFAULT_ANGLE_DELTA_CDF.iter().enumerate() {
            let mode = V_PRED + i;
            let row = ctx.angle_delta_cdf(mode).unwrap();
            assert_eq!(row.len(), (2 * MAX_ANGLE_DELTA + 1) + 1);
            assert_eq!(row, want);
        }

        // DC_PRED (0, below V_PRED) is non-directional → None.
        assert!(ctx.angle_delta_cdf(0).is_none());
        // SMOOTH_PRED (9 == V_PRED + DIRECTIONAL_MODES) and beyond → None.
        assert!(ctx.angle_delta_cdf(V_PRED + DIRECTIONAL_MODES).is_none());
        assert!(ctx
            .angle_delta_cdf(V_PRED + DIRECTIONAL_MODES + 3)
            .is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through an
    /// angle-delta default CDF selected by the §8.3.2 selection (a
    /// directional `YMode`, here `D45_PRED == V_PRED + 2`), confirming the
    /// chosen row matches the §9.4 source, the decoded symbol lands in
    /// `0..(2 * MAX_ANGLE_DELTA + 1)` and the working copy adapts.
    #[test]
    fn decode_angle_delta_through_default_cdf() {
        // D45_PRED = 3 → angle-delta row 3 - V_PRED = 2.
        let mode = V_PRED + 2;
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.angle_delta;

        let row = tile_ctx.angle_delta_cdf(mode).unwrap();
        assert_eq!(row, &DEFAULT_ANGLE_DELTA_CDF[2]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < (2 * MAX_ANGLE_DELTA + 1),
            "angle_delta must code a symbol in 0..(2 * MAX_ANGLE_DELTA + 1)"
        );
        assert_ne!(
            tile_ctx.angle_delta, before,
            "read_symbol must adapt the working CDF"
        );
        assert_eq!(DEFAULT_ANGLE_DELTA_CDF[2][0], 3780);
    }

    // -----------------------------------------------------------------
    // Round 136 — coefficient-token entry sub-group tests.
    // -----------------------------------------------------------------

    /// §3 coefficient-entry constants are pinned to their spec values.
    #[test]
    fn coeff_entry_constants_pinned() {
        assert_eq!(PLANE_TYPES, 2);
        assert_eq!(TX_SIZES, 5);
        assert_eq!(COEFF_CDF_Q_CTXS, 4);
        assert_eq!(TXB_SKIP_CONTEXTS, 13);
        assert_eq!(EOB_COEF_CONTEXTS, 9);
        assert_eq!(DC_SIGN_CONTEXTS, 3);
    }

    /// Every coefficient-entry default table carries the declared §9.4
    /// dimensions, and every cumulative-frequency row is well-formed: the
    /// last data entry is `1 << 15 == 32768`, the trailing adaptation
    /// counter is `0`, and the cumulative frequencies strictly increase.
    #[test]
    fn coeff_entry_default_tables_well_formed() {
        fn check_row(row: &[u16]) {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768: {row:?}");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0: {row:?}");
            for w in row[..n].windows(2) {
                assert!(w[0] < w[1], "cdf must strictly increase: {row:?}");
            }
        }

        // Txb_Skip: [4][5][13][3].
        assert_eq!(DEFAULT_TXB_SKIP_CDF.len(), COEFF_CDF_Q_CTXS);
        for q in &DEFAULT_TXB_SKIP_CDF {
            assert_eq!(q.len(), TX_SIZES);
            for tx in q {
                assert_eq!(tx.len(), TXB_SKIP_CONTEXTS);
                for row in tx {
                    check_row(row);
                }
            }
        }

        // Eob_Pt_{16..256}: [4][2][2][N]; Eob_Pt_{512,1024}: [4][2][N].
        macro_rules! check_eob_two {
            ($t:expr, $inner:expr) => {{
                assert_eq!($t.len(), COEFF_CDF_Q_CTXS);
                for q in &$t {
                    assert_eq!(q.len(), PLANE_TYPES);
                    for pt in q {
                        assert_eq!(pt.len(), 2);
                        for row in pt {
                            assert_eq!(row.len(), $inner);
                            check_row(row);
                        }
                    }
                }
            }};
        }
        check_eob_two!(DEFAULT_EOB_PT_16_CDF, 6);
        check_eob_two!(DEFAULT_EOB_PT_32_CDF, 7);
        check_eob_two!(DEFAULT_EOB_PT_64_CDF, 8);
        check_eob_two!(DEFAULT_EOB_PT_128_CDF, 9);
        check_eob_two!(DEFAULT_EOB_PT_256_CDF, 10);

        macro_rules! check_eob_one {
            ($t:expr, $inner:expr) => {{
                assert_eq!($t.len(), COEFF_CDF_Q_CTXS);
                for q in &$t {
                    assert_eq!(q.len(), PLANE_TYPES);
                    for row in q {
                        assert_eq!(row.len(), $inner);
                        check_row(row);
                    }
                }
            }};
        }
        check_eob_one!(DEFAULT_EOB_PT_512_CDF, 11);
        check_eob_one!(DEFAULT_EOB_PT_1024_CDF, 12);

        // Eob_Extra: [4][5][2][9][3].
        assert_eq!(DEFAULT_EOB_EXTRA_CDF.len(), COEFF_CDF_Q_CTXS);
        for q in &DEFAULT_EOB_EXTRA_CDF {
            assert_eq!(q.len(), TX_SIZES);
            for tx in q {
                assert_eq!(tx.len(), PLANE_TYPES);
                for pt in tx {
                    assert_eq!(pt.len(), EOB_COEF_CONTEXTS);
                    for row in pt {
                        check_row(row);
                    }
                }
            }
        }

        // Dc_Sign: [4][2][3][3].
        assert_eq!(DEFAULT_DC_SIGN_CDF.len(), COEFF_CDF_Q_CTXS);
        for q in &DEFAULT_DC_SIGN_CDF {
            assert_eq!(q.len(), PLANE_TYPES);
            for pt in q {
                assert_eq!(pt.len(), DC_SIGN_CONTEXTS);
                for row in pt {
                    check_row(row);
                }
            }
        }
    }

    /// Spot-check the §9.4 coefficient-entry default values byte-for-byte.
    /// A mis-keyed digit during transcription breaks the equality.
    #[test]
    fn coeff_entry_default_byte_exact_values() {
        // Txb_Skip first/last anchors (q0,tx0,ctx0) and (q0,tx0,ctx12).
        assert_eq!(DEFAULT_TXB_SKIP_CDF[0][0][0], [31849, 32768, 0]);
        assert_eq!(DEFAULT_TXB_SKIP_CDF[0][0][12], [32117, 32768, 0]);
        // A non-trivial interior row (q3, tx4, ctx6 — the last triplet of
        // the table, a placeholder 16384 context).
        assert_eq!(DEFAULT_TXB_SKIP_CDF[3][4][12], [16384, 32768, 0]);

        // Eob_Pt_16: first (q0,pt0,inter0) and last (q3,pt1,inter1) rows.
        assert_eq!(
            DEFAULT_EOB_PT_16_CDF[0][0][0],
            [840, 1039, 1980, 4895, 32768, 0]
        );
        assert_eq!(
            DEFAULT_EOB_PT_16_CDF[3][1][1],
            [7297, 10767, 19273, 28194, 32768, 0]
        );

        // Eob_Pt_1024: longest EOB-position rows (12 wide).
        assert_eq!(
            DEFAULT_EOB_PT_1024_CDF[0][0],
            [393, 421, 751, 1623, 3160, 6352, 13345, 18047, 22571, 25830, 32768, 0]
        );

        // Eob_Extra first anchor.
        assert_eq!(DEFAULT_EOB_EXTRA_CDF[0][0][0][0], [16961, 32768, 0]);

        // Dc_Sign: the `128 * N` fixed-point form, identical across the
        // four q-contexts in the spec.
        assert_eq!(DEFAULT_DC_SIGN_CDF[0][0][0], [128 * 125, 32768, 0]);
        assert_eq!(DEFAULT_DC_SIGN_CDF[0][1][2], [128 * 135, 32768, 0]);
        assert_eq!(DEFAULT_DC_SIGN_CDF[3][0][0], DEFAULT_DC_SIGN_CDF[0][0][0]);
        assert_eq!(128 * 125u16, 16000);
    }

    /// §8.3.1 `init_coeff_cdfs` q-context derivation from `base_q_idx`,
    /// covering each boundary (`<=20`, `<=60`, `<=120`, else).
    #[test]
    fn coeff_cdf_q_ctx_mapping() {
        assert_eq!(coeff_cdf_q_ctx(0), 0);
        assert_eq!(coeff_cdf_q_ctx(20), 0);
        assert_eq!(coeff_cdf_q_ctx(21), 1);
        assert_eq!(coeff_cdf_q_ctx(60), 1);
        assert_eq!(coeff_cdf_q_ctx(61), 2);
        assert_eq!(coeff_cdf_q_ctx(120), 2);
        assert_eq!(coeff_cdf_q_ctx(121), 3);
        assert_eq!(coeff_cdf_q_ctx(255), 3);
        // Every result is a valid COEFF_CDF_Q_CTXS index.
        for q in 0u8..=255 {
            assert!(coeff_cdf_q_ctx(q) < COEFF_CDF_Q_CTXS);
        }
    }

    /// §8.3.1 `init_coeff_cdfs`: each working coefficient-entry array is
    /// set to a copy of the `Default_*_Cdf[ idx ]` slice for the derived
    /// `idx`, and mutating the working copy leaves the §9.4 source intact.
    #[test]
    fn coeff_entry_init_coeff_cdfs_selects_q_ctx() {
        // A fresh context is seeded from idx 0.
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.txb_skip, DEFAULT_TXB_SKIP_CDF[0]);
        assert_eq!(ctx.dc_sign, DEFAULT_DC_SIGN_CDF[0]);

        // init_coeff_cdfs re-selects the slice for a high base_q_idx.
        ctx.init_coeff_cdfs(200); // -> idx 3
        assert_eq!(ctx.txb_skip, DEFAULT_TXB_SKIP_CDF[3]);
        assert_eq!(ctx.eob_pt_16, DEFAULT_EOB_PT_16_CDF[3]);
        assert_eq!(ctx.eob_pt_32, DEFAULT_EOB_PT_32_CDF[3]);
        assert_eq!(ctx.eob_pt_64, DEFAULT_EOB_PT_64_CDF[3]);
        assert_eq!(ctx.eob_pt_128, DEFAULT_EOB_PT_128_CDF[3]);
        assert_eq!(ctx.eob_pt_256, DEFAULT_EOB_PT_256_CDF[3]);
        assert_eq!(ctx.eob_pt_512, DEFAULT_EOB_PT_512_CDF[3]);
        assert_eq!(ctx.eob_pt_1024, DEFAULT_EOB_PT_1024_CDF[3]);
        assert_eq!(ctx.eob_extra, DEFAULT_EOB_EXTRA_CDF[3]);
        assert_eq!(ctx.dc_sign, DEFAULT_DC_SIGN_CDF[3]);

        // And for a mid base_q_idx.
        ctx.init_coeff_cdfs(50); // -> idx 1
        assert_eq!(ctx.txb_skip, DEFAULT_TXB_SKIP_CDF[1]);
        assert_eq!(ctx.eob_extra, DEFAULT_EOB_EXTRA_CDF[1]);

        // Mutating the working copy leaves the §9.4 source untouched.
        ctx.txb_skip_cdf(0, 0).unwrap()[0] = 12345;
        assert_ne!(ctx.txb_skip[0][0], DEFAULT_TXB_SKIP_CDF[1][0][0]);
        assert_eq!(DEFAULT_TXB_SKIP_CDF[1][0][0], [30371, 32768, 0]);
    }

    /// §8.3.2 selectors return the correct §9.4 rows and reject
    /// out-of-range indices.
    #[test]
    fn coeff_entry_selectors_return_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults(); // idx 0

        assert_eq!(
            ctx.txb_skip_cdf(2, 5).unwrap(),
            &DEFAULT_TXB_SKIP_CDF[0][2][5]
        );
        assert!(ctx.txb_skip_cdf(TX_SIZES, 0).is_none());
        assert!(ctx.txb_skip_cdf(0, TXB_SKIP_CONTEXTS).is_none());

        assert_eq!(
            ctx.eob_pt_16_cdf(1, 1).unwrap(),
            &DEFAULT_EOB_PT_16_CDF[0][1][1]
        );
        assert_eq!(
            ctx.eob_pt_32_cdf(0, 0).unwrap(),
            &DEFAULT_EOB_PT_32_CDF[0][0][0]
        );
        assert_eq!(
            ctx.eob_pt_64_cdf(1, 0).unwrap(),
            &DEFAULT_EOB_PT_64_CDF[0][1][0]
        );
        assert_eq!(
            ctx.eob_pt_128_cdf(0, 1).unwrap(),
            &DEFAULT_EOB_PT_128_CDF[0][0][1]
        );
        assert_eq!(
            ctx.eob_pt_256_cdf(1, 1).unwrap(),
            &DEFAULT_EOB_PT_256_CDF[0][1][1]
        );
        assert!(ctx.eob_pt_16_cdf(PLANE_TYPES, 0).is_none());
        assert!(ctx.eob_pt_256_cdf(0, 2).is_none());

        assert_eq!(
            ctx.eob_pt_512_cdf(1).unwrap(),
            &DEFAULT_EOB_PT_512_CDF[0][1]
        );
        assert_eq!(
            ctx.eob_pt_1024_cdf(0).unwrap(),
            &DEFAULT_EOB_PT_1024_CDF[0][0]
        );
        assert!(ctx.eob_pt_512_cdf(PLANE_TYPES).is_none());

        assert_eq!(
            ctx.eob_extra_cdf(4, 1, 8).unwrap(),
            &DEFAULT_EOB_EXTRA_CDF[0][4][1][8]
        );
        assert!(ctx.eob_extra_cdf(TX_SIZES, 0, 0).is_none());
        assert!(ctx.eob_extra_cdf(0, 0, EOB_COEF_CONTEXTS).is_none());

        assert_eq!(
            ctx.dc_sign_cdf(1, 2).unwrap(),
            &DEFAULT_DC_SIGN_CDF[0][1][2]
        );
        assert!(ctx.dc_sign_cdf(PLANE_TYPES, 0).is_none());
        assert!(ctx.dc_sign_cdf(0, DC_SIGN_CONTEXTS).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// coefficient-entry default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the decoded
    /// symbol is in range, and the working copy adapts.
    #[test]
    fn decode_coeff_entry_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];

        // txb_skip (all_zero): a binary symbol.
        {
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut ctx = TileCdfContext::new_from_defaults();
            let before = ctx.txb_skip;
            let row = ctx.txb_skip_cdf(0, 0).unwrap();
            assert_eq!(row, &DEFAULT_TXB_SKIP_CDF[0][0][0]);
            let sym = dec.read_symbol(row).unwrap();
            assert!((sym as usize) < 2, "all_zero must code 0 or 1");
            assert_ne!(ctx.txb_skip, before, "read_symbol must adapt the CDF");
        }

        // eob_pt_16: a 5-class symbol.
        {
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut ctx = TileCdfContext::new_from_defaults();
            let before = ctx.eob_pt_16;
            let row = ctx.eob_pt_16_cdf(0, 0).unwrap();
            assert_eq!(row, &DEFAULT_EOB_PT_16_CDF[0][0][0]);
            let sym = dec.read_symbol(row).unwrap();
            assert!((sym as usize) < 5, "eob_pt_16 codes 0..5");
            assert_ne!(ctx.eob_pt_16, before, "read_symbol must adapt the CDF");
        }

        // dc_sign: a binary symbol from the `128 * N` defaults.
        {
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut ctx = TileCdfContext::new_from_defaults();
            let before = ctx.dc_sign;
            let row = ctx.dc_sign_cdf(0, 0).unwrap();
            assert_eq!(row, &DEFAULT_DC_SIGN_CDF[0][0][0]);
            let sym = dec.read_symbol(row).unwrap();
            assert!((sym as usize) < 2, "dc_sign must code 0 or 1");
            assert_ne!(ctx.dc_sign, before, "read_symbol must adapt the CDF");
        }
    }

    // -----------------------------------------------------------------
    // Round 138 — `coeff_base_eob` sub-group tests.
    // -----------------------------------------------------------------

    /// §3 `SIG_COEF_CONTEXTS_EOB` pins to its spec value.
    #[test]
    fn coeff_base_eob_constants_pinned() {
        assert_eq!(SIG_COEF_CONTEXTS_EOB, 4);
    }

    /// `Default_Coeff_Base_Eob_Cdf` carries the declared §9.4 dimensions
    /// and every cumulative-frequency row is well-formed: the last data
    /// entry is `1 << 15 == 32768`, the trailing adaptation counter is
    /// `0`, and the cumulative frequencies strictly increase.
    #[test]
    fn coeff_base_eob_default_table_well_formed() {
        fn check_row(row: &[u16]) {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768: {row:?}");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0: {row:?}");
            for w in row[..n].windows(2) {
                assert!(w[0] < w[1], "cdf must strictly increase: {row:?}");
            }
        }

        assert_eq!(DEFAULT_COEFF_BASE_EOB_CDF.len(), COEFF_CDF_Q_CTXS);
        for q in &DEFAULT_COEFF_BASE_EOB_CDF {
            assert_eq!(q.len(), TX_SIZES);
            for tx in q {
                assert_eq!(tx.len(), PLANE_TYPES);
                for pt in tx {
                    assert_eq!(pt.len(), SIG_COEF_CONTEXTS_EOB);
                    for row in pt {
                        assert_eq!(row.len(), 4);
                        check_row(row);
                    }
                }
            }
        }
    }

    /// Spot-check the §9.4 `Default_Coeff_Base_Eob_Cdf` initial values
    /// byte-for-byte. A mis-keyed digit during transcription breaks the
    /// equality. Anchors include the (q0, tx0) first / last context and
    /// the flat-distribution placeholder row that pads every q-context's
    /// last `(txSz, chroma)` slice.
    #[test]
    fn coeff_base_eob_default_byte_exact_values() {
        // (q0, tx0, pt0): first context and last context of the row.
        assert_eq!(
            DEFAULT_COEFF_BASE_EOB_CDF[0][0][0][0],
            [17837, 29055, 32768, 0]
        );
        assert_eq!(
            DEFAULT_COEFF_BASE_EOB_CDF[0][0][0][3],
            [24926, 28948, 32768, 0]
        );

        // (q0, tx0, pt1): chroma plane.
        assert_eq!(
            DEFAULT_COEFF_BASE_EOB_CDF[0][0][1][0],
            [21365, 30026, 32768, 0]
        );

        // Every q-context's (tx4, pt1) slice is the same flat
        // {10923, 21845, 32768, 0} placeholder for all four ctx values
        // (the §9.4 sentinel for an unreachable chroma row at the
        // largest TX size).
        let flat = [10923u16, 21845, 32768, 0];
        for (q, q_slice) in DEFAULT_COEFF_BASE_EOB_CDF.iter().enumerate() {
            for (ctx, row) in q_slice[4][1].iter().enumerate() {
                assert_eq!(
                    *row, flat,
                    "(q={q}, tx=4, pt=1, ctx={ctx}) must be the flat placeholder"
                );
            }
        }

        // (q3, tx3, pt1, ctx3): a non-trivial interior anchor — the
        // last (luma, ctx) row of the second-largest TX size at the
        // highest q-context.
        assert_eq!(
            DEFAULT_COEFF_BASE_EOB_CDF[3][3][1][3],
            [31767, 32712, 32768, 0]
        );

        // (q3, tx4, pt0, ctx0): the largest TX size, luma, first ctx.
        assert_eq!(
            DEFAULT_COEFF_BASE_EOB_CDF[3][4][0][0],
            [12358, 24977, 32768, 0]
        );
    }

    /// §8.3.1 `init_coeff_cdfs`: the working `coeff_base_eob` array is
    /// set to a copy of `Default_Coeff_Base_Eob_Cdf[ idx ]` for the
    /// derived `idx`, and mutating the working copy leaves the §9.4
    /// source intact.
    #[test]
    fn coeff_base_eob_init_coeff_cdfs_selects_q_ctx() {
        // A fresh context is seeded from idx 0.
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.coeff_base_eob, DEFAULT_COEFF_BASE_EOB_CDF[0]);

        // init_coeff_cdfs re-selects the slice for a high base_q_idx.
        ctx.init_coeff_cdfs(200); // -> idx 3
        assert_eq!(ctx.coeff_base_eob, DEFAULT_COEFF_BASE_EOB_CDF[3]);

        // And for a mid base_q_idx.
        ctx.init_coeff_cdfs(50); // -> idx 1
        assert_eq!(ctx.coeff_base_eob, DEFAULT_COEFF_BASE_EOB_CDF[1]);

        // Mutating the working copy leaves the §9.4 source untouched.
        ctx.coeff_base_eob_cdf(0, 0, 0).unwrap()[0] = 12345;
        assert_ne!(
            ctx.coeff_base_eob[0][0][0],
            DEFAULT_COEFF_BASE_EOB_CDF[1][0][0][0]
        );
        assert_eq!(
            DEFAULT_COEFF_BASE_EOB_CDF[1][0][0][0],
            [17560, 29888, 32768, 0]
        );
    }

    /// §8.3.2 `coeff_base_eob` selector returns the correct §9.4 row
    /// for the in-range `(txSzCtx, ptype, ctx)` triple and rejects each
    /// out-of-range axis.
    #[test]
    fn coeff_base_eob_selector_returns_default_row() {
        let mut ctx = TileCdfContext::new_from_defaults(); // idx 0

        // In-range: (tx=2, pt=1, ctx=2).
        assert_eq!(
            ctx.coeff_base_eob_cdf(2, 1, 2).unwrap(),
            &DEFAULT_COEFF_BASE_EOB_CDF[0][2][1][2]
        );
        // In-range: (tx=4, pt=0, ctx=0).
        assert_eq!(
            ctx.coeff_base_eob_cdf(4, 0, 0).unwrap(),
            &DEFAULT_COEFF_BASE_EOB_CDF[0][4][0][0]
        );

        // Out-of-range on each axis.
        assert!(ctx.coeff_base_eob_cdf(TX_SIZES, 0, 0).is_none());
        assert!(ctx.coeff_base_eob_cdf(0, PLANE_TYPES, 0).is_none());
        assert!(ctx
            .coeff_base_eob_cdf(0, 0, SIG_COEF_CONTEXTS_EOB)
            .is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through the
    /// `coeff_base_eob` default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the decoded
    /// symbol is in range (`0..3`, since the base level is
    /// `coeff_base_eob + 1` and the spec restricts it to 1, 2, or 3),
    /// and the working copy adapts.
    #[test]
    fn decode_coeff_base_eob_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];

        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.coeff_base_eob;
        let row = ctx.coeff_base_eob_cdf(0, 0, 0).unwrap();
        assert_eq!(row, &DEFAULT_COEFF_BASE_EOB_CDF[0][0][0][0]);
        let sym = dec.read_symbol(row).unwrap();
        assert!((sym as usize) < 3, "coeff_base_eob codes 0..3");
        assert_ne!(
            ctx.coeff_base_eob, before,
            "read_symbol must adapt the working CDF"
        );
    }

    // -----------------------------------------------------------------
    // Round 139 — `coeff_base` sub-group tests.
    // -----------------------------------------------------------------

    /// §3 `SIG_COEF_CONTEXTS` pins to its spec value (42).
    #[test]
    fn coeff_base_constants_pinned() {
        assert_eq!(SIG_COEF_CONTEXTS, 42);
    }

    /// `Default_Coeff_Base_Cdf` shape + per-row well-formedness: every
    /// 5-entry row is `[c0, c1, c2, 32768, 0]` with `c0 < c1 < c2 < 32768`
    /// (and a fresh §8.3 adaptation counter at the tail).
    #[test]
    fn coeff_base_default_table_well_formed() {
        fn check_row(row: &[u16]) {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768: {row:?}");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0: {row:?}");
            for w in row[..n].windows(2) {
                assert!(w[0] < w[1], "cdf must strictly increase: {row:?}");
            }
        }

        assert_eq!(DEFAULT_COEFF_BASE_CDF.len(), COEFF_CDF_Q_CTXS);
        for q in &DEFAULT_COEFF_BASE_CDF {
            assert_eq!(q.len(), TX_SIZES);
            for tx in q {
                assert_eq!(tx.len(), PLANE_TYPES);
                for pt in tx {
                    assert_eq!(pt.len(), SIG_COEF_CONTEXTS);
                    for row in pt {
                        assert_eq!(row.len(), 5);
                        check_row(row);
                    }
                }
            }
        }
    }

    /// Spot-check the §9.4 `Default_Coeff_Base_Cdf` initial values
    /// byte-for-byte. A mis-keyed digit during transcription breaks
    /// the equality. Anchors include the (q0, tx0, pt0) first row, a
    /// chroma row at the same q-ctx, an interior anchor, and the
    /// flat-distribution placeholder that pads the largest TX size's
    /// chroma slice across every q-context.
    #[test]
    fn coeff_base_default_byte_exact_values() {
        // (q0, tx0, pt0, ctx0): first row of the table.
        assert_eq!(
            DEFAULT_COEFF_BASE_CDF[0][0][0][0],
            [4034, 8930, 12727, 32768, 0]
        );
        // (q0, tx0, pt1, ctx0): chroma plane, same TX size.
        assert_eq!(
            DEFAULT_COEFF_BASE_CDF[0][0][1][0],
            [6302, 16444, 21761, 32768, 0]
        );
        // (q1, tx0, pt0, ctx0): next q-context, luma.
        assert_eq!(
            DEFAULT_COEFF_BASE_CDF[1][0][0][0],
            [6041, 11854, 15927, 32768, 0]
        );
        // (q3, tx4, pt0, ctx0): largest TX size, luma, highest q-ctx.
        assert_eq!(
            DEFAULT_COEFF_BASE_CDF[3][4][0][0],
            [4137, 10847, 15682, 32768, 0]
        );

        // Every q-context's (tx4, pt1) slice is the same flat
        // {8192, 16384, 24576, 32768, 0} placeholder for all 42 ctx
        // values (§9.4 sentinel for the unreachable largest-TX
        // chroma row, exactly mirroring the r138 (tx4, pt1)
        // placeholder pattern in `Default_Coeff_Base_Eob_Cdf`).
        let flat = [8192u16, 16384, 24576, 32768, 0];
        for (q, q_slice) in DEFAULT_COEFF_BASE_CDF.iter().enumerate() {
            for (ctx, row) in q_slice[4][1].iter().enumerate() {
                assert_eq!(
                    *row, flat,
                    "(q={q}, tx=4, pt=1, ctx={ctx}) must be the flat placeholder"
                );
            }
        }
    }

    /// §8.3.1 `init_coeff_cdfs`: the working `coeff_base` array is
    /// set to a copy of `Default_Coeff_Base_Cdf[ idx ]` for the
    /// derived `idx`, and mutating the working copy leaves the §9.4
    /// source intact.
    #[test]
    fn coeff_base_init_coeff_cdfs_selects_q_ctx() {
        // A fresh context is seeded from idx 0.
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.coeff_base, DEFAULT_COEFF_BASE_CDF[0]);

        // init_coeff_cdfs re-selects the slice for a high base_q_idx.
        ctx.init_coeff_cdfs(200); // -> idx 3
        assert_eq!(ctx.coeff_base, DEFAULT_COEFF_BASE_CDF[3]);

        // And for a mid base_q_idx.
        ctx.init_coeff_cdfs(50); // -> idx 1
        assert_eq!(ctx.coeff_base, DEFAULT_COEFF_BASE_CDF[1]);

        // Mutating the working copy leaves the §9.4 source untouched.
        ctx.coeff_base_cdf(0, 0, 0).unwrap()[0] = 12345;
        assert_ne!(ctx.coeff_base[0][0][0], DEFAULT_COEFF_BASE_CDF[1][0][0][0]);
        assert_eq!(
            DEFAULT_COEFF_BASE_CDF[1][0][0][0],
            [6041, 11854, 15927, 32768, 0]
        );
    }

    /// §8.3.2 `coeff_base` selector returns the correct §9.4 row
    /// for the in-range `(txSzCtx, ptype, ctx)` triple and rejects
    /// each out-of-range axis. Probes the boundary contexts at
    /// `SIG_COEF_CONTEXTS - 1` and the §3 `SIG_COEF_CONTEXTS_2D`
    /// split point.
    #[test]
    fn coeff_base_selector_returns_default_row() {
        let mut ctx = TileCdfContext::new_from_defaults(); // idx 0

        // In-range: (tx=2, pt=1, ctx=10).
        assert_eq!(
            ctx.coeff_base_cdf(2, 1, 10).unwrap(),
            &DEFAULT_COEFF_BASE_CDF[0][2][1][10]
        );
        // In-range at the SIG_COEF_CONTEXTS_2D split point: (tx=1, pt=0, ctx=26).
        assert_eq!(
            ctx.coeff_base_cdf(1, 0, 26).unwrap(),
            &DEFAULT_COEFF_BASE_CDF[0][1][0][26]
        );
        // In-range at the highest context: (tx=0, pt=0, ctx=SIG_COEF_CONTEXTS-1).
        assert_eq!(
            ctx.coeff_base_cdf(0, 0, SIG_COEF_CONTEXTS - 1).unwrap(),
            &DEFAULT_COEFF_BASE_CDF[0][0][0][SIG_COEF_CONTEXTS - 1]
        );

        // Out-of-range on each axis.
        assert!(ctx.coeff_base_cdf(TX_SIZES, 0, 0).is_none());
        assert!(ctx.coeff_base_cdf(0, PLANE_TYPES, 0).is_none());
        assert!(ctx.coeff_base_cdf(0, 0, SIG_COEF_CONTEXTS).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through the
    /// `coeff_base` default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the
    /// decoded symbol is in range (`0..4`, since `coeff_base` codes
    /// the four-symbol alphabet `0..3`), and the working copy adapts.
    #[test]
    fn decode_coeff_base_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];

        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.coeff_base;
        let row = ctx.coeff_base_cdf(0, 0, 0).unwrap();
        assert_eq!(row, &DEFAULT_COEFF_BASE_CDF[0][0][0][0]);
        let sym = dec.read_symbol(row).unwrap();
        assert!((sym as usize) < 4, "coeff_base codes 0..4");
        assert_ne!(
            ctx.coeff_base, before,
            "read_symbol must adapt the working CDF"
        );
    }

    // -----------------------------------------------------------------
    // Round 140 — `coeff_br` sub-group tests.
    // -----------------------------------------------------------------

    /// §3 `LEVEL_CONTEXTS` / `BR_CDF_SIZE` pin to their spec values
    /// (21 and 4 respectively).
    #[test]
    fn coeff_br_constants_pinned() {
        assert_eq!(LEVEL_CONTEXTS, 21);
        assert_eq!(BR_CDF_SIZE, 4);
    }

    /// `Default_Coeff_Br_Cdf` shape + per-row well-formedness: every
    /// 5-entry row is `[c0, c1, c2, 32768, 0]` with `c0 < c1 < c2 < 32768`
    /// (and a fresh §8.3 adaptation counter at the tail).
    #[test]
    fn coeff_br_default_table_well_formed() {
        fn check_row(row: &[u16]) {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768: {row:?}");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0: {row:?}");
            for w in row[..n].windows(2) {
                assert!(w[0] < w[1], "cdf must strictly increase: {row:?}");
            }
        }

        assert_eq!(DEFAULT_COEFF_BR_CDF.len(), COEFF_CDF_Q_CTXS);
        for q in &DEFAULT_COEFF_BR_CDF {
            assert_eq!(q.len(), TX_SIZES);
            for tx in q {
                assert_eq!(tx.len(), PLANE_TYPES);
                for pt in tx {
                    assert_eq!(pt.len(), LEVEL_CONTEXTS);
                    for row in pt {
                        assert_eq!(row.len(), BR_CDF_SIZE + 1);
                        check_row(row);
                    }
                }
            }
        }
    }

    /// Spot-check the §9.4 `Default_Coeff_Br_Cdf` initial values
    /// byte-for-byte. Anchors include the (q0, tx0, pt0) first row,
    /// the chroma plane at the same q-ctx, an interior anchor, and
    /// the flat-distribution placeholder that pads the largest TX
    /// size across every q-context.
    #[test]
    fn coeff_br_default_byte_exact_values() {
        // (q0, tx0, pt0, ctx0): first row of the table.
        assert_eq!(
            DEFAULT_COEFF_BR_CDF[0][0][0][0],
            [14298, 20718, 24174, 32768, 0]
        );
        // (q0, tx0, pt1, ctx0): chroma plane, same TX size.
        assert_eq!(
            DEFAULT_COEFF_BR_CDF[0][0][1][0],
            [15967, 22905, 26286, 32768, 0]
        );
        // (q0, tx0, pt0, ctx20 = last LEVEL_CONTEXTS row of (q0, tx0, pt0)).
        assert_eq!(
            DEFAULT_COEFF_BR_CDF[0][0][0][LEVEL_CONTEXTS - 1],
            [3034, 5860, 8462, 32768, 0]
        );
        // Largest TX size, chroma slice across every q-context, should be
        // the flat {8192, 16384, 24576, 32768, 0} sentinel for the
        // unreachable rows (§9.4 placeholder pattern, identical to
        // r138 / r139).
        let flat = [8192u16, 16384, 24576, 32768, 0];
        for (q, q_slice) in DEFAULT_COEFF_BR_CDF.iter().enumerate() {
            for (ctx, row) in q_slice[4][1].iter().enumerate() {
                assert_eq!(
                    *row, flat,
                    "(q={q}, tx=4, pt=1, ctx={ctx}) must be the flat placeholder"
                );
            }
        }
    }

    /// §8.3.1 `init_coeff_cdfs`: the working `coeff_br` array is set to
    /// a copy of `Default_Coeff_Br_Cdf[ idx ]` for the derived `idx`,
    /// and mutating the working copy leaves the §9.4 source intact.
    #[test]
    fn coeff_br_init_coeff_cdfs_selects_q_ctx() {
        // A fresh context is seeded from idx 0.
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.coeff_br, DEFAULT_COEFF_BR_CDF[0]);

        // init_coeff_cdfs re-selects the slice for a high base_q_idx.
        ctx.init_coeff_cdfs(200); // -> idx 3
        assert_eq!(ctx.coeff_br, DEFAULT_COEFF_BR_CDF[3]);

        // And for a mid base_q_idx.
        ctx.init_coeff_cdfs(50); // -> idx 1
        assert_eq!(ctx.coeff_br, DEFAULT_COEFF_BR_CDF[1]);

        // Mutating the working copy leaves the §9.4 source untouched.
        ctx.coeff_br_cdf(0, 0, 0).unwrap()[0] = 12345;
        assert_ne!(ctx.coeff_br[0][0][0], DEFAULT_COEFF_BR_CDF[1][0][0][0]);
    }

    /// §8.3.2 `coeff_br` selector returns the correct §9.4 row for the
    /// in-range `(txSzCtx, ptype, ctx)` triple, clamps `txSzCtx` to
    /// `TX_32X32 = 3` per the spec selector
    /// `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]`,
    /// and rejects each out-of-range axis other than `txSzCtx`.
    #[test]
    fn coeff_br_selector_returns_default_row_and_clamps_tx() {
        let mut ctx = TileCdfContext::new_from_defaults(); // idx 0
        const TX_32X32: usize = 3;

        // In-range: (tx=2, pt=1, ctx=10).
        assert_eq!(
            ctx.coeff_br_cdf(2, 1, 10).unwrap(),
            &DEFAULT_COEFF_BR_CDF[0][2][1][10]
        );
        // At the largest unclamped index: (tx=TX_32X32, pt=0, ctx=20).
        assert_eq!(
            ctx.coeff_br_cdf(TX_32X32, 0, LEVEL_CONTEXTS - 1).unwrap(),
            &DEFAULT_COEFF_BR_CDF[0][TX_32X32][0][LEVEL_CONTEXTS - 1]
        );
        // Clamped: (tx=TX_SIZES-1, pt=0, ctx=0) reads the TX_32X32 slot.
        assert_eq!(
            ctx.coeff_br_cdf(TX_SIZES - 1, 0, 0).unwrap(),
            &DEFAULT_COEFF_BR_CDF[0][TX_32X32][0][0]
        );
        // An out-of-spec txSzCtx (above TX_SIZES) is still accepted; the
        // clamp pins the lookup at TX_32X32.
        assert_eq!(
            ctx.coeff_br_cdf(99, 1, 5).unwrap(),
            &DEFAULT_COEFF_BR_CDF[0][TX_32X32][1][5]
        );

        // Out-of-range on the ptype / ctx axes is rejected.
        assert!(ctx.coeff_br_cdf(0, PLANE_TYPES, 0).is_none());
        assert!(ctx.coeff_br_cdf(0, 0, LEVEL_CONTEXTS).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through the
    /// `coeff_br` default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the
    /// decoded symbol is in range (`0..BR_CDF_SIZE`), and the working
    /// copy adapts.
    #[test]
    fn decode_coeff_br_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];

        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.coeff_br;
        let row = ctx.coeff_br_cdf(0, 0, 0).unwrap();
        assert_eq!(row, &DEFAULT_COEFF_BR_CDF[0][0][0][0]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < BR_CDF_SIZE,
            "coeff_br codes 0..BR_CDF_SIZE"
        );
        assert_ne!(
            ctx.coeff_br, before,
            "read_symbol must adapt the working CDF"
        );
    }

    // -----------------------------------------------------------------
    // §8.3.2 get_coeff_base_ctx / get_br_ctx neighbour derivation.
    // Expected ctx values are hand-computed from the §8.3.2 pseudocode.
    // -----------------------------------------------------------------

    /// §3 constants the §8.3.2 helpers depend on pin to their spec values.
    #[test]
    fn coeff_ctx_constants_pin() {
        assert_eq!(SIG_COEF_CONTEXTS_2D, 26);
        assert_eq!(SIG_REF_DIFF_OFFSET_NUM, 5);
        assert_eq!(NUM_BASE_LEVELS, 2);
        assert_eq!(COEFF_BASE_RANGE, 12);
        assert_eq!(TX_SIZES_ALL, 19);
        assert_eq!(TX_CLASS_2D, 0);
        assert_eq!(TX_CLASS_HORIZ, 1);
        assert_eq!(TX_CLASS_VERT, 2);
        // Each TX_SIZES_ALL-indexed table has the right shape.
        assert_eq!(ADJUSTED_TX_SIZE.len(), TX_SIZES_ALL);
        assert_eq!(TX_WIDTH.len(), TX_SIZES_ALL);
        assert_eq!(TX_HEIGHT.len(), TX_SIZES_ALL);
        assert_eq!(TX_WIDTH_LOG2.len(), TX_SIZES_ALL);
        assert_eq!(COEFF_BASE_CTX_OFFSET.len(), TX_SIZES_ALL);
        // Coeff_Base_Pos_Ctx_Offset[ idx ] = SIG_COEF_CONTEXTS_2D + 5*idx.
        assert_eq!(COEFF_BASE_POS_CTX_OFFSET, [26, 31, 36]);
        // Width / log2 self-consistency: Tx_Width == 1 << Tx_Width_Log2.
        for t in 0..TX_SIZES_ALL {
            assert_eq!(TX_WIDTH[t], 1 << TX_WIDTH_LOG2[t]);
        }
    }

    /// §8.3.2 `get_tx_class()` reduces the directional flags to the three
    /// transform classes.
    #[test]
    fn tx_class_reduction() {
        // No directional flag -> 2D.
        assert_eq!(
            get_tx_class(false, false, false, false, false, false),
            TX_CLASS_2D
        );
        // Any vertical flag -> VERT (vertical takes precedence in spec).
        assert_eq!(
            get_tx_class(true, false, false, false, false, false),
            TX_CLASS_VERT
        );
        assert_eq!(
            get_tx_class(false, true, false, false, false, false),
            TX_CLASS_VERT
        );
        assert_eq!(
            get_tx_class(false, false, true, false, false, false),
            TX_CLASS_VERT
        );
        // Horizontal flags (no vertical) -> HORIZ.
        assert_eq!(
            get_tx_class(false, false, false, true, false, false),
            TX_CLASS_HORIZ
        );
        assert_eq!(
            get_tx_class(false, false, false, false, true, false),
            TX_CLASS_HORIZ
        );
        assert_eq!(
            get_tx_class(false, false, false, false, false, true),
            TX_CLASS_HORIZ
        );
    }

    /// §8.3.2 `get_coeff_base_ctx(..., isEob=1)` buckets for TX_4X4
    /// (bwl=2, height=4 -> (height<<bwl)=16, /8=2, /4=4).
    #[test]
    fn coeff_base_eob_buckets() {
        let quant = [0i32; 16];
        // c == 0 -> SIG_COEF_CONTEXTS - 4.
        assert_eq!(
            get_coeff_base_ctx(&quant, 0, TX_CLASS_2D, 0, 0, true),
            SIG_COEF_CONTEXTS - 4
        );
        // c <= 2 -> SIG_COEF_CONTEXTS - 3.
        assert_eq!(
            get_coeff_base_ctx(&quant, 0, TX_CLASS_2D, 0, 1, true),
            SIG_COEF_CONTEXTS - 3
        );
        assert_eq!(
            get_coeff_base_ctx(&quant, 0, TX_CLASS_2D, 0, 2, true),
            SIG_COEF_CONTEXTS - 3
        );
        // 2 < c <= 4 -> SIG_COEF_CONTEXTS - 2.
        assert_eq!(
            get_coeff_base_ctx(&quant, 0, TX_CLASS_2D, 0, 3, true),
            SIG_COEF_CONTEXTS - 2
        );
        assert_eq!(
            get_coeff_base_ctx(&quant, 0, TX_CLASS_2D, 0, 4, true),
            SIG_COEF_CONTEXTS - 2
        );
        // c > 4 -> SIG_COEF_CONTEXTS - 1.
        assert_eq!(
            get_coeff_base_ctx(&quant, 0, TX_CLASS_2D, 0, 5, true),
            SIG_COEF_CONTEXTS - 1
        );

        // The coeff_base_eob ctx wrapper reduces onto 0..SIG_COEF_CONTEXTS_EOB.
        assert_eq!(get_coeff_base_eob_ctx(&quant, 0, TX_CLASS_2D, 0, 0), 0);
        assert_eq!(get_coeff_base_eob_ctx(&quant, 0, TX_CLASS_2D, 0, 1), 1);
        assert_eq!(get_coeff_base_eob_ctx(&quant, 0, TX_CLASS_2D, 0, 3), 2);
        assert_eq!(get_coeff_base_eob_ctx(&quant, 0, TX_CLASS_2D, 0, 5), 3);
        // The reduced ctx must address a coeff_base_eob CDF row.
        let mut tile = TileCdfContext::new_from_defaults();
        let eob_ctx = get_coeff_base_eob_ctx(&quant, 0, TX_CLASS_2D, 0, 5);
        assert!(tile.coeff_base_eob_cdf(0, 0, eob_ctx).is_some());
    }

    /// §8.3.2 2D branch early return: `row == 0 && col == 0` -> ctx 0.
    #[test]
    fn coeff_base_2d_dc_returns_zero() {
        let quant = [9i32; 64]; // neighbours irrelevant for the DC return.
        assert_eq!(get_coeff_base_ctx(&quant, 1, TX_CLASS_2D, 0, 0, false), 0);
    }

    /// §8.3.2 2D branch with a non-trivial Coeff_Base_Ctx_Offset entry.
    /// TX_8X8 (bwl=3), pos=9 -> row=1, col=1, offset[1][1][1]=6. A single
    /// neighbour magnitude 2 gives mag=2, ctx=Min((2+1)>>1,4)=1 -> 1+6=7.
    #[test]
    fn coeff_base_2d_with_neighbour_mag() {
        let mut quant = [0i32; 64];
        // Sig_Ref_Diff_Offset[2D][0] = (0,1): refRow=1, refCol=2 ->
        // Quant[(1<<3)+2] = Quant[10].
        quant[10] = 2;
        let ctx = get_coeff_base_ctx(&quant, 1, TX_CLASS_2D, 9, 0, false);
        assert_eq!(ctx, 7);
        assert!(ctx < SIG_COEF_CONTEXTS);

        // Saturated 2D neighbourhood: all five offsets carry magnitude 3
        // -> mag=15, ctx=Min((15+1)>>1,4)=Min(8,4)=4. 4 + offset[1][1][1]=6
        // -> 10.
        let mut sat = [0i32; 64];
        for &(r, c) in &[(1usize, 2usize), (2, 1), (2, 2), (1, 3), (3, 1)] {
            sat[(r << 3) + c] = 3;
        }
        let ctx_sat = get_coeff_base_ctx(&sat, 1, TX_CLASS_2D, 9, 0, false);
        assert_eq!(ctx_sat, 10);
        assert!(ctx_sat < SIG_COEF_CONTEXTS);
    }

    /// §8.3.2 vertical branch: idx=row, Coeff_Base_Pos_Ctx_Offset[Min(row,2)].
    /// TX_8X8 (bwl=3), pos=24 -> row=3, col=0; Min(3,2)=2 -> offset 36.
    /// Two saturated neighbours give mag=6, ctx=Min((6+1)>>1,4)=3 -> 39.
    #[test]
    fn coeff_base_vert_branch() {
        let mut quant = [0i32; 64];
        // Sig_Ref_Diff_Offset[VERT][0]=(0,1): refRow=3,refCol=1 -> Quant[25].
        // Sig_Ref_Diff_Offset[VERT][1]=(1,0): refRow=4,refCol=0 -> Quant[32].
        quant[25] = 3;
        quant[32] = 3;
        let ctx = get_coeff_base_ctx(&quant, 1, TX_CLASS_VERT, 24, 0, false);
        assert_eq!(ctx, 39);
        assert!(ctx < SIG_COEF_CONTEXTS);
        assert!(ctx >= SIG_COEF_CONTEXTS_2D, "1D tail uses the >=26 range");
    }

    /// §8.3.2 horizontal branch with Min(col,2) clamping.
    /// TX_8X8 (bwl=3). pos=3 -> row=0,col=3; Min(3,2)=2 -> offset 36.
    /// Quant[4]=1 -> mag=1, ctx=Min((1+1)>>1,4)=1 -> 37.
    #[test]
    fn coeff_base_horiz_branch_clamped() {
        let mut quant = [0i32; 64];
        // Sig_Ref_Diff_Offset[HORIZ][0]=(0,1): refRow=0,refCol=4 -> Quant[4].
        quant[4] = 1;
        let ctx = get_coeff_base_ctx(&quant, 1, TX_CLASS_HORIZ, 3, 0, false);
        assert_eq!(ctx, 37);
        assert!(ctx < SIG_COEF_CONTEXTS);

        // col=1 -> idx not clamped: Coeff_Base_Pos_Ctx_Offset[1] = 31.
        // pos=1 -> row=0,col=1. Empty neighbourhood -> mag=0, ctx=0 -> 31.
        let empty = [0i32; 64];
        let ctx2 = get_coeff_base_ctx(&empty, 1, TX_CLASS_HORIZ, 1, 0, false);
        assert_eq!(ctx2, 31);
    }

    /// §8.3.2 `get_br_ctx`: pos==0 path returns the magnitude bucket alone.
    /// TX_4X4 (bwl=2,txw=4,txh=4). Quant[1]=5 -> mag=Min(5,15)=5,
    /// mag=Min((5+1)>>1,6)=3. pos==0 -> ctx=3.
    #[test]
    fn br_ctx_pos_zero() {
        let mut quant = [0i32; 16];
        // Mag_Ref_Offset[2D][0]=(0,1): refRow=0,refCol=1 -> Quant[0*4+1]=Quant[1].
        quant[1] = 5;
        let ctx = get_br_ctx(&quant, 0, TX_CLASS_2D, 0);
        assert_eq!(ctx, 3);
        assert!(ctx < LEVEL_CONTEXTS);
    }

    /// §8.3.2 `get_br_ctx` magnitude clamp: each neighbour saturates at
    /// COEFF_BASE_RANGE+NUM_BASE_LEVELS+1 = 15; three at 100 -> mag=45,
    /// Min((45+1)>>1,6)=6. TX_4X4 pos==0 -> ctx=6 (the max bucket).
    #[test]
    fn br_ctx_saturated_clamp() {
        let mut quant = [0i32; 16];
        // 2D offsets from (0,0): (0,1)->Q[1], (1,0)->Q[4], (1,1)->Q[5].
        quant[1] = 100;
        quant[4] = 100;
        quant[5] = 100;
        let ctx = get_br_ctx(&quant, 0, TX_CLASS_2D, 0);
        assert_eq!(ctx, 6);
        assert!(ctx < LEVEL_CONTEXTS);
    }

    /// §8.3.2 `get_br_ctx` 2D branch: inner (row<2 && col<2) -> +7,
    /// outer -> +14. TX_8X8 (bwl=3).
    #[test]
    fn br_ctx_2d_inner_outer() {
        // Inner: pos=9 -> row=1,col=1. Quant[10]=1 (offset (0,1) -> col 2)
        // -> mag=1, Min((1+1)>>1,6)=1, +7 -> 8.
        let mut inner = [0i32; 64];
        inner[10] = 1; // refRow=1,refCol=2 -> 1*8+2 = 10.
        let c_inner = get_br_ctx(&inner, 1, TX_CLASS_2D, 9);
        assert_eq!(c_inner, 8);
        assert!(c_inner < LEVEL_CONTEXTS);

        // Outer: pos=18 -> row=2,col=2. Empty neighbourhood -> mag=0,
        // Min((0+1)>>1,6)=0, +14 -> 14.
        let empty = [0i32; 64];
        let c_outer = get_br_ctx(&empty, 1, TX_CLASS_2D, 18);
        assert_eq!(c_outer, 14);
        assert!(c_outer < LEVEL_CONTEXTS);
    }

    /// §8.3.2 `get_br_ctx` horizontal branch (txClass==1): col==0 -> +7,
    /// else +14. TX_8X8 (bwl=3), empty neighbourhood -> mag bucket 0.
    #[test]
    fn br_ctx_horiz_branch() {
        let empty = [0i32; 64];
        // col==0: pos=8 -> row=1,col=0 -> mag bucket 0, +7 -> 7.
        assert_eq!(get_br_ctx(&empty, 1, TX_CLASS_HORIZ, 8), 7);
        // col!=0: pos=1 -> row=0,col=1 -> +14 -> 14.
        assert_eq!(get_br_ctx(&empty, 1, TX_CLASS_HORIZ, 1), 14);
    }

    /// §8.3.2 `get_br_ctx` vertical branch (txClass==2 / else): row==0 ->
    /// +7, else +14. TX_8X8 (bwl=3), empty neighbourhood -> mag bucket 0.
    #[test]
    fn br_ctx_vert_branch() {
        let empty = [0i32; 64];
        // row==0: pos=1 -> row=0,col=1 -> +7 -> 7.
        assert_eq!(get_br_ctx(&empty, 1, TX_CLASS_VERT, 1), 7);
        // row!=0: pos=8 -> row=1,col=0 -> +14 -> 14.
        assert_eq!(get_br_ctx(&empty, 1, TX_CLASS_VERT, 8), 14);

        // The derived ctx must address a coeff_br CDF row.
        let mut tile = TileCdfContext::new_from_defaults();
        let ctx = get_br_ctx(&empty, 1, TX_CLASS_VERT, 1);
        assert!(tile.coeff_br_cdf(1, 0, ctx).is_some());
    }

    // -----------------------------------------------------------------
    // Round 142 — §5.11.40 `compute_tx_type()` derivation.
    // -----------------------------------------------------------------

    /// `Tx_Size_Sqr_Up[ TX_SIZES_ALL ]` (§ Additional tables): each row
    /// entry is a valid `TX_4X4..TX_64X64` ordinal, and the equality
    /// `Tx_Width[ Tx_Size_Sqr_Up[ t ] ] == Tx_Height[ Tx_Size_Sqr_Up[ t ] ]`
    /// holds because the destination is always a square size. Spot-checks
    /// the rectangular entries lift to `Max(w, h)`-sided squares.
    #[test]
    fn tx_size_sqr_up_table_is_well_formed() {
        assert_eq!(TX_SIZE_SQR_UP.len(), TX_SIZES_ALL);
        for &t in &TX_SIZE_SQR_UP {
            assert!(
                t <= TX_64X64,
                "Tx_Size_Sqr_Up entry must be a square tx-size ordinal"
            );
            assert_eq!(
                TX_WIDTH[t], TX_HEIGHT[t],
                "Tx_Size_Sqr_Up destination must be square"
            );
        }
        // §6.10.16 ordinals: 5 = TX_4X8, 7 = TX_8X16, 13 = TX_4X16 —
        // each lifts to the Max(w,h)-sided square per the spec.
        assert_eq!(TX_SIZE_SQR_UP[5], TX_8X8); // 4x8 -> 8x8
        assert_eq!(TX_SIZE_SQR_UP[7], TX_16X16); // 8x16 -> 16x16
        assert_eq!(TX_SIZE_SQR_UP[13], TX_16X16); // 4x16 -> 16x16
        assert_eq!(TX_SIZE_SQR_UP[11], TX_64X64); // 32x64 -> 64x64
    }

    /// §6.10.16 / §3 transform-size and transform-type ordinals match
    /// the spec-listed values; the ordinal constants are the same
    /// values previously used as locally-scoped `const TX_*` shadows.
    #[test]
    fn tx_size_and_tx_type_ordinals_match_spec() {
        // §6.10.16 TxSize semantics table rows 0..4.
        assert_eq!(TX_4X4, 0);
        assert_eq!(TX_8X8, 1);
        assert_eq!(TX_16X16, 2);
        assert_eq!(TX_32X32, 3);
        assert_eq!(TX_64X64, 4);
        // §6.10.19 TxType enumeration rows 0..15.
        assert_eq!(DCT_DCT, 0);
        assert_eq!(ADST_DCT, 1);
        assert_eq!(DCT_ADST, 2);
        assert_eq!(ADST_ADST, 3);
        assert_eq!(FLIPADST_DCT, 4);
        assert_eq!(DCT_FLIPADST, 5);
        assert_eq!(FLIPADST_FLIPADST, 6);
        assert_eq!(ADST_FLIPADST, 7);
        assert_eq!(FLIPADST_ADST, 8);
        assert_eq!(IDTX, 9);
        assert_eq!(V_DCT, 10);
        assert_eq!(H_DCT, 11);
        assert_eq!(V_ADST, 12);
        assert_eq!(H_ADST, 13);
        assert_eq!(V_FLIPADST, 14);
        assert_eq!(H_FLIPADST, 15);
        // §3 TX_SET_TYPES_* counts match the table-row counts.
        assert_eq!(TX_SET_TYPES_INTRA, TX_TYPE_IN_SET_INTRA.len());
        assert_eq!(TX_SET_TYPES_INTER, TX_TYPE_IN_SET_INTER.len());
    }

    /// `Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]` (§ Additional tables):
    /// per-row spot-check + length pin. Each entry lies in `0..TX_TYPES`
    /// (so [`is_tx_type_in_set`] never returns the out-of-range `false`).
    #[test]
    fn mode_to_txfm_table_matches_spec() {
        assert_eq!(MODE_TO_TXFM.len(), UV_INTRA_MODES_CFL_ALLOWED);
        for &t in &MODE_TO_TXFM {
            assert!(t < TX_TYPES);
        }
        // Spec § Additional tables (Mode_To_Txfm), per-row:
        assert_eq!(MODE_TO_TXFM[0], DCT_DCT); // DC_PRED
        assert_eq!(MODE_TO_TXFM[1], ADST_DCT); // V_PRED
        assert_eq!(MODE_TO_TXFM[2], DCT_ADST); // H_PRED
        assert_eq!(MODE_TO_TXFM[4], ADST_ADST); // D135_PRED
        assert_eq!(MODE_TO_TXFM[9], ADST_ADST); // SMOOTH_PRED
        assert_eq!(MODE_TO_TXFM[12], ADST_ADST); // PAETH_PRED
        assert_eq!(MODE_TO_TXFM[13], DCT_DCT); // UV_CFL_PRED
    }

    /// §5.11.40 `is_tx_type_in_set` over the four inter `txSet` rows.
    /// `TX_SET_DCTONLY` admits only `DCT_DCT`; `TX_SET_INTER_1` admits
    /// all 16; `TX_SET_INTER_2` admits 0..=11 only; `TX_SET_INTER_3`
    /// admits `{ IDTX, DCT_DCT }`.
    #[test]
    fn is_tx_type_in_set_inter_per_row() {
        // TX_SET_DCTONLY (0): DCT_DCT only.
        assert!(is_tx_type_in_set(true, TX_SET_DCTONLY, DCT_DCT));
        assert!(!is_tx_type_in_set(true, TX_SET_DCTONLY, ADST_DCT));
        assert!(!is_tx_type_in_set(true, TX_SET_DCTONLY, IDTX));
        // TX_SET_INTER_1 (1): every type admitted.
        for t in 0..TX_TYPES {
            assert!(
                is_tx_type_in_set(true, TX_SET_INTER_1, t),
                "INTER_1 admits {t}"
            );
        }
        // TX_SET_INTER_2 (2): 0..=11 yes, 12..=15 no.
        for t in 0..12 {
            assert!(
                is_tx_type_in_set(true, TX_SET_INTER_2, t),
                "INTER_2 admits {t}"
            );
        }
        for t in 12..16 {
            assert!(
                !is_tx_type_in_set(true, TX_SET_INTER_2, t),
                "INTER_2 rejects {t}"
            );
        }
        // TX_SET_INTER_3 (3): IDTX + DCT_DCT only.
        assert!(is_tx_type_in_set(true, TX_SET_INTER_3, DCT_DCT));
        assert!(is_tx_type_in_set(true, TX_SET_INTER_3, IDTX));
        assert!(!is_tx_type_in_set(true, TX_SET_INTER_3, ADST_DCT));
        assert!(!is_tx_type_in_set(true, TX_SET_INTER_3, H_FLIPADST));
    }

    /// §5.11.40 `is_tx_type_in_set` over the three intra `txSet` rows.
    /// `TX_SET_INTRA_1` admits 4 cross-DCT/ADST types + IDTX + V_DCT +
    /// H_DCT (i.e. 7 entries); `TX_SET_INTRA_2` drops V_DCT + H_DCT (5
    /// entries). Out-of-range tx_set / tx_type returns `false`.
    #[test]
    fn is_tx_type_in_set_intra_per_row() {
        // TX_SET_DCTONLY: DCT_DCT only.
        assert!(is_tx_type_in_set(false, TX_SET_DCTONLY, DCT_DCT));
        assert!(!is_tx_type_in_set(false, TX_SET_DCTONLY, IDTX));
        // TX_SET_INTRA_1: 4 cross-set + { IDTX, V_DCT, H_DCT }.
        for t in [DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST, IDTX, V_DCT, H_DCT] {
            assert!(
                is_tx_type_in_set(false, TX_SET_INTRA_1, t),
                "INTRA_1 admits {t}"
            );
        }
        for t in [FLIPADST_DCT, DCT_FLIPADST, V_ADST, H_FLIPADST] {
            assert!(
                !is_tx_type_in_set(false, TX_SET_INTRA_1, t),
                "INTRA_1 rejects {t}"
            );
        }
        // TX_SET_INTRA_2: drops V_DCT / H_DCT vs INTRA_1.
        assert!(is_tx_type_in_set(false, TX_SET_INTRA_2, IDTX));
        assert!(!is_tx_type_in_set(false, TX_SET_INTRA_2, V_DCT));
        assert!(!is_tx_type_in_set(false, TX_SET_INTRA_2, H_DCT));
        // Out-of-range inputs return false rather than panicking.
        assert!(!is_tx_type_in_set(false, 99, DCT_DCT));
        assert!(!is_tx_type_in_set(true, 99, DCT_DCT));
        assert!(!is_tx_type_in_set(false, TX_SET_INTRA_1, TX_TYPES));
    }

    /// §5.11.40 `compute_tx_type` short-circuits: `Lossless` returns
    /// `DCT_DCT` ignoring everything; `txSzSqrUp > TX_32X32` (e.g.
    /// `TX_64X64`, ordinal 4) returns `DCT_DCT` regardless of plane /
    /// is_inter / tx_set.
    #[test]
    fn compute_tx_type_lossless_and_large_fallback() {
        let tx_types = |_: u32, _: u32| ADST_ADST; // would-be luma result
                                                   // Lossless: must collapse to DCT_DCT even for plane 0.
        assert_eq!(
            compute_tx_type(
                0,
                TX_8X8,
                true,
                false,
                TX_SET_INTRA_1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                tx_types
            ),
            DCT_DCT
        );
        // TX_64X64 -> txSzSqrUp == TX_64X64 > TX_32X32 -> DCT_DCT.
        assert_eq!(
            compute_tx_type(
                0,
                TX_64X64,
                false,
                true,
                TX_SET_INTER_1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                tx_types
            ),
            DCT_DCT
        );
        // TX_32X64 (ordinal 11) also lifts to TX_64X64.
        assert_eq!(
            compute_tx_type(
                0,
                11,
                false,
                true,
                TX_SET_INTER_1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                tx_types
            ),
            DCT_DCT
        );
        // Out-of-range tx_sz: treat as fallback (no panic).
        assert_eq!(
            compute_tx_type(
                0,
                99,
                false,
                true,
                TX_SET_INTER_1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                tx_types
            ),
            DCT_DCT
        );
    }

    /// §5.11.40 `compute_tx_type` luma branch (`plane == 0`): returns
    /// the `TxTypes[blockY][blockX]` cache entry verbatim (no
    /// admission filter, no `Max(MiRow, ...)` lift). The chroma
    /// `is_tx_type_in_set` admission filter does not apply.
    #[test]
    fn compute_tx_type_plane_zero_returns_cache_entry() {
        // The closure encodes a 2D cache; the function must read
        // exactly (block_y, block_x) without applying subsampling /
        // MI-grid lift.
        let cache = |y: u32, x: u32| ((y * 4 + x) as usize) % TX_TYPES;
        let result = compute_tx_type(
            0,
            TX_8X8,
            false,
            true,
            TX_SET_INTER_1,
            10,
            20, // MiRow / MiCol (ignored on luma)
            3,
            2,
            1,
            1, // subsampling (ignored on luma)
            5, // uv_mode (ignored on luma)
            cache,
        );
        assert_eq!(result, (2 * 4 + 3) % TX_TYPES);
    }

    /// §5.11.40 `compute_tx_type` inter chroma branch: reads
    /// `TxTypes[Max(MiRow, blockY<<ssY)][Max(MiCol, blockX<<ssX)]`
    /// then admits per `is_tx_type_in_set(true, txSet, txType)` else
    /// returns `DCT_DCT`. Covers (a) the `Max` lift firing when
    /// `MiCol > blockX<<ssX`, (b) the admission filter passing, and
    /// (c) the admission filter falling back.
    #[test]
    fn compute_tx_type_inter_chroma_max_lift_and_admission() {
        // (a) Max lift: subsampling_x = 1, block_x = 1, MiCol = 4. The
        // spec computes `Max(MiCol, 1 << 1) = Max(4, 2) = 4`, so the
        // cache lookup must be at column 4 (not 2). Encode the cache
        // so column-4 reads return ADST_DCT but column-2 reads return
        // a clearly-disadmitted ADST_FLIPADST.
        let cache = |y: u32, x: u32| {
            if y == 4 && x == 4 {
                ADST_DCT // admitted under TX_SET_INTER_1 / INTER_2
            } else if y == 4 && x == 2 {
                ADST_FLIPADST // not selected if Max lift works
            } else {
                FLIPADST_FLIPADST
            }
        };
        let r = compute_tx_type(
            1,
            TX_8X8,
            false,
            true,
            TX_SET_INTER_1,
            4,
            4, // MiRow = MiCol = 4
            1,
            1,
            1,
            1, // block_x = block_y = 1; subsampling = 1
            0,
            cache,
        );
        assert_eq!(r, ADST_DCT);

        // (b) Admission pass: TX_SET_INTER_3 admits { IDTX, DCT_DCT }.
        // Cache returns IDTX; result must be IDTX (not DCT_DCT).
        let idtx_cache = |_: u32, _: u32| IDTX;
        let r2 = compute_tx_type(
            2,
            TX_8X8,
            false,
            true,
            TX_SET_INTER_3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            idtx_cache,
        );
        assert_eq!(r2, IDTX);

        // (c) Admission fail: TX_SET_INTER_3 rejects ADST_DCT, so the
        // §5.11.40 `!is_tx_type_in_set` fallback returns DCT_DCT.
        let adst_cache = |_: u32, _: u32| ADST_DCT;
        let r3 = compute_tx_type(
            2,
            TX_8X8,
            false,
            true,
            TX_SET_INTER_3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            adst_cache,
        );
        assert_eq!(r3, DCT_DCT);
    }

    /// §5.11.40 `compute_tx_type` intra chroma branch:
    /// `Mode_To_Txfm[UVMode]` then `is_tx_type_in_set(false, txSet,
    /// txType)`. The `TxTypes` closure is *not* invoked. Covers a
    /// successful admission path, a fallback path, and an
    /// out-of-range `uv_mode` (returns DCT_DCT).
    #[test]
    fn compute_tx_type_intra_chroma_uv_mode_path() {
        // The closure must NOT be invoked on the intra chroma branch
        // — drive it through a panic-on-call closure to prove it.
        let never_call = |_: u32, _: u32| -> usize {
            panic!("luma TxTypes cache must not be read on intra-chroma branch")
        };

        // (a) UVMode = D135_PRED (4) -> ADST_ADST. INTRA_1 admits it.
        let r1 = compute_tx_type(
            1,
            TX_8X8,
            false,
            false,
            TX_SET_INTRA_1,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            never_call,
        );
        assert_eq!(r1, ADST_ADST);

        // (b) UVMode = V_PRED (1) -> ADST_DCT. Under TX_SET_DCTONLY
        // the admission filter rejects -> DCT_DCT fallback.
        let r2 = compute_tx_type(
            1,
            TX_8X8,
            false,
            false,
            TX_SET_DCTONLY,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            never_call,
        );
        assert_eq!(r2, DCT_DCT);

        // (c) Out-of-range uv_mode -> DCT_DCT.
        let r3 = compute_tx_type(
            2,
            TX_8X8,
            false,
            false,
            TX_SET_INTRA_1,
            0,
            0,
            0,
            0,
            0,
            0,
            99,
            never_call,
        );
        assert_eq!(r3, DCT_DCT);
    }

    /// §5.11.40 `compute_tx_type` and the existing
    /// [`inter_tx_type_set`] / [`intra_tx_type_set`] selectors form a
    /// closed loop: the txSet that selector returns for a given
    /// (tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set) triple is the txSet
    /// the admission filter must use. Spot-check the loop for a
    /// 16×16 inter chroma block.
    #[test]
    fn compute_tx_type_uses_selector_consistent_tx_set() {
        // tx_sz = TX_16X16 -> sqr = sqr_up = TX_16X16.
        let tx_sz_sqr = TX_WIDTH[TX_16X16].trailing_zeros() - 2; // 16 -> log2=4, bwl_index 2
                                                                 // Just use the spec's explicit ordinal: TX_16X16.
        let tx_set = inter_tx_type_set(TX_16X16 as u32, TX_16X16 as u32, false);
        assert_eq!(tx_set, TX_SET_INTER_2);

        // Cache returns DCT_DCT (admitted under INTER_2). Result is
        // DCT_DCT.
        let cache = |_: u32, _: u32| DCT_DCT;
        let r = compute_tx_type(1, TX_16X16, false, true, tx_set, 0, 0, 0, 0, 0, 0, 0, cache);
        assert_eq!(r, DCT_DCT);
        // Silence "unused" warning on the local — it documents the
        // bwl ordinal the closure-free path computes.
        let _ = tx_sz_sqr;
    }

    // -----------------------------------------------------------------
    // Round 143 — inter-intra group tests.
    // -----------------------------------------------------------------

    /// §3 / §6.10.27: `INTERINTRA_MODES = 4`, matching the spec's
    /// enumeration count.
    #[test]
    fn interintra_modes_constant_matches_spec() {
        assert_eq!(INTERINTRA_MODES, 4);
    }

    /// `Default_Inter_Intra_Cdf` / `Default_Inter_Intra_Mode_Cdf` /
    /// `Default_Wedge_Inter_Intra_Cdf` are pinned verbatim from §9.4 and
    /// satisfy the §8.2.6 well-formedness invariants every other §9.4
    /// table satisfies: the trailing entry of each row is `0` (the §8.3
    /// adaptation counter) and the next-to-last is `1 << 15 == 32768`.
    #[test]
    fn inter_intra_default_tables_pinned() {
        // Shape pins straight off the §9.4 listing.
        assert_eq!(DEFAULT_INTER_INTRA_CDF.len(), BLOCK_SIZE_GROUPS - 1);
        assert_eq!(DEFAULT_INTER_INTRA_MODE_CDF.len(), BLOCK_SIZE_GROUPS - 1);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF.len(), BLOCK_SIZES);

        // Spec-pinned values: `Default_Inter_Intra_Cdf` (verbatim from
        // §9.4 listing, p.434).
        assert_eq!(DEFAULT_INTER_INTRA_CDF[0], [26887, 32768, 0]);
        assert_eq!(DEFAULT_INTER_INTRA_CDF[1], [27597, 32768, 0]);
        assert_eq!(DEFAULT_INTER_INTRA_CDF[2], [30237, 32768, 0]);

        // `Default_Inter_Intra_Mode_Cdf` (verbatim from §9.4, p.434).
        assert_eq!(
            DEFAULT_INTER_INTRA_MODE_CDF[0],
            [1875, 11082, 27332, 32768, 0]
        );
        assert_eq!(
            DEFAULT_INTER_INTRA_MODE_CDF[1],
            [2473, 9996, 26388, 32768, 0]
        );
        assert_eq!(
            DEFAULT_INTER_INTRA_MODE_CDF[2],
            [4238, 11537, 25926, 32768, 0]
        );

        // `Default_Wedge_Inter_Intra_Cdf` (verbatim from §9.4, p.436):
        // per the spec note, only indices 3..=9 are reachable; rows 0..2
        // and 10..21 are placeholder `{16384, 32768, 0}`.
        for row in &DEFAULT_WEDGE_INTER_INTRA_CDF[0..3] {
            assert_eq!(row, &[16384, 32768, 0]);
        }
        for row in &DEFAULT_WEDGE_INTER_INTRA_CDF[10..BLOCK_SIZES] {
            assert_eq!(row, &[16384, 32768, 0]);
        }
        // Reachable band (BLOCK_8X8..=BLOCK_32X32 -> indices 3..=9).
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[3], [20036, 32768, 0]);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[4], [24957, 32768, 0]);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[5], [26704, 32768, 0]);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[6], [27530, 32768, 0]);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[7], [29564, 32768, 0]);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[8], [29444, 32768, 0]);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[9], [26872, 32768, 0]);

        // §8.2.6 well-formedness invariants on every row of every CDF.
        for row in &DEFAULT_INTER_INTRA_CDF {
            assert_eq!(row[row.len() - 1], 0);
            assert_eq!(row[row.len() - 2], 1 << 15);
        }
        for row in &DEFAULT_INTER_INTRA_MODE_CDF {
            assert_eq!(row[row.len() - 1], 0);
            assert_eq!(row[row.len() - 2], 1 << 15);
        }
        for row in &DEFAULT_WEDGE_INTER_INTRA_CDF {
            assert_eq!(row[row.len() - 1], 0);
            assert_eq!(row[row.len() - 2], 1 << 15);
        }
    }

    /// §8.3.1 `init_non_coeff_cdfs`: the working-copy fields are seeded
    /// verbatim from the §9.4 defaults.
    #[test]
    fn inter_intra_init_from_defaults() {
        let ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.inter_intra, DEFAULT_INTER_INTRA_CDF);
        assert_eq!(ctx.inter_intra_mode, DEFAULT_INTER_INTRA_MODE_CDF);
        assert_eq!(ctx.wedge_inter_intra, DEFAULT_WEDGE_INTER_INTRA_CDF);
    }

    /// `interintra_ctx` implements the §8.3.2
    /// `ctx = Size_Group[ MiSize ] - 1` mapping across the entire
    /// `BLOCK_SIZES` axis: rows where `Size_Group[ MiSize ] == 0` (i.e.
    /// outside the §5.11.28 syntax gate) return `None`, the rest return
    /// `Size_Group[ MiSize ] - 1` in `0..(BLOCK_SIZE_GROUPS - 1)`.
    #[test]
    fn interintra_ctx_matches_size_group_minus_one() {
        for (mi_size, &g) in SIZE_GROUP.iter().enumerate() {
            match interintra_ctx(mi_size) {
                None => assert_eq!(g, 0),
                Some(ctx) => {
                    assert!(g > 0);
                    assert_eq!(ctx, g - 1);
                    assert!(ctx < BLOCK_SIZE_GROUPS - 1);
                }
            }
        }
        // Out-of-range MiSize is rejected.
        assert!(interintra_ctx(BLOCK_SIZES).is_none());
        assert!(interintra_ctx(BLOCK_SIZES + 7).is_none());

        // §5.11.28 syntax gate band: BLOCK_8X8 (3) .. BLOCK_32X32 (9)
        // inclusive must all return Some. Spot-check each.
        for mi_size in 3..=9 {
            assert!(
                interintra_ctx(mi_size).is_some(),
                "MiSize {mi_size} (in BLOCK_8X8..=BLOCK_32X32) must yield Some(ctx)"
            );
        }
    }

    /// §8.3.2 inter-intra selectors return the right §9.4 rows for every
    /// valid context, and reject out-of-range indices with `None`.
    #[test]
    fn inter_intra_selectors_return_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();

        for (i, want) in DEFAULT_INTER_INTRA_CDF.iter().enumerate() {
            let row = ctx.inter_intra_cdf(i).unwrap();
            assert_eq!(row.len(), 3);
            assert_eq!(row, want);
        }
        assert!(ctx.inter_intra_cdf(BLOCK_SIZE_GROUPS - 1).is_none());
        assert!(ctx.inter_intra_cdf(BLOCK_SIZE_GROUPS).is_none());

        for (i, want) in DEFAULT_INTER_INTRA_MODE_CDF.iter().enumerate() {
            let row = ctx.inter_intra_mode_cdf(i).unwrap();
            assert_eq!(row.len(), INTERINTRA_MODES + 1);
            assert_eq!(row, want);
        }
        assert!(ctx.inter_intra_mode_cdf(BLOCK_SIZE_GROUPS - 1).is_none());

        for (i, want) in DEFAULT_WEDGE_INTER_INTRA_CDF.iter().enumerate() {
            let row = ctx.wedge_inter_intra_cdf(i).unwrap();
            assert_eq!(row.len(), 3);
            assert_eq!(row, want);
        }
        assert!(ctx.wedge_inter_intra_cdf(BLOCK_SIZES).is_none());
        assert!(ctx.wedge_inter_intra_cdf(BLOCK_SIZES + 5).is_none());
    }

    /// Working-copy independence: mutating the context via the selectors
    /// must not touch the §9.4 source tables.
    #[test]
    fn inter_intra_working_copy_is_independent() {
        let mut ctx = TileCdfContext::new_from_defaults();
        ctx.inter_intra_cdf(0).unwrap()[0] = 12345;
        ctx.inter_intra_mode_cdf(1).unwrap()[0] = 23456;
        ctx.wedge_inter_intra_cdf(3).unwrap()[0] = 34567;
        assert_ne!(ctx.inter_intra[0][0], DEFAULT_INTER_INTRA_CDF[0][0]);
        assert_ne!(
            ctx.inter_intra_mode[1][0],
            DEFAULT_INTER_INTRA_MODE_CDF[1][0]
        );
        assert_ne!(
            ctx.wedge_inter_intra[3][0],
            DEFAULT_WEDGE_INTER_INTRA_CDF[3][0]
        );
        // Spec sources unchanged.
        assert_eq!(DEFAULT_INTER_INTRA_CDF[0][0], 26887);
        assert_eq!(DEFAULT_INTER_INTRA_MODE_CDF[1][0], 2473);
        assert_eq!(DEFAULT_WEDGE_INTER_INTRA_CDF[3][0], 20036);
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through both an
    /// `interintra` (binary) and an `interintra_mode` default CDF
    /// selected via the §8.3.2 `ctx = Size_Group[ MiSize ] - 1` mapping
    /// for a `BLOCK_16X16` (`MiSize = 6`, `Size_Group = 2`, `ctx = 1`),
    /// confirming the row matches the §9.4 source, the decode lands in
    /// range, and the working copy adapts.
    #[test]
    fn decode_inter_intra_through_default_cdf() {
        // BLOCK_16X16 (MiSize = 6) -> Size_Group[6] = 2 -> ctx = 1.
        assert_eq!(SIZE_GROUP[6], 2);
        let ctx_val = interintra_ctx(6).unwrap();
        assert_eq!(ctx_val, 1);

        // Binary `interintra` decode.
        {
            let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut tile_ctx = TileCdfContext::new_from_defaults();
            let before = tile_ctx.inter_intra;

            let row = tile_ctx.inter_intra_cdf(ctx_val).unwrap();
            assert_eq!(row, &DEFAULT_INTER_INTRA_CDF[1]);
            let sym = dec.read_symbol(row).unwrap();
            assert!(sym < 2, "interintra must code a binary symbol");

            assert_ne!(
                tile_ctx.inter_intra, before,
                "read_symbol must adapt the working CDF"
            );
        }

        // 4-symbol `interintra_mode` decode.
        {
            let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
            let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
            let mut tile_ctx = TileCdfContext::new_from_defaults();
            let before = tile_ctx.inter_intra_mode;

            let row = tile_ctx.inter_intra_mode_cdf(ctx_val).unwrap();
            assert_eq!(row, &DEFAULT_INTER_INTRA_MODE_CDF[1]);
            let sym = dec.read_symbol(row).unwrap();
            assert!(
                (sym as usize) < INTERINTRA_MODES,
                "interintra_mode must code a symbol in 0..INTERINTRA_MODES"
            );

            assert_ne!(
                tile_ctx.inter_intra_mode, before,
                "read_symbol must adapt the working CDF"
            );
        }
    }

    /// End-to-end: drive the §8.2 `SymbolDecoder` through a
    /// `wedge_interintra` default CDF for a `BLOCK_16X16` (`MiSize = 6`,
    /// inside the §5.11.28 reachable band), confirming the row matches
    /// the §9.4 source, the binary decode lands in range, and the working
    /// copy adapts.
    #[test]
    fn decode_wedge_inter_intra_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.wedge_inter_intra;

        // MiSize = 6 (BLOCK_16X16) is inside the §5.11.28 reachable band
        // (rows 3..=9) where the §9.4 table holds non-placeholder values.
        let row = tile_ctx.wedge_inter_intra_cdf(6).unwrap();
        assert_eq!(row, &DEFAULT_WEDGE_INTER_INTRA_CDF[6]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(sym < 2, "wedge_interintra must code a binary symbol");

        assert_ne!(
            tile_ctx.wedge_inter_intra, before,
            "read_symbol must adapt the working CDF"
        );
    }

    // -----------------------------------------------------------------
    // Round 144 — wedge-index CDF tests.
    // -----------------------------------------------------------------

    /// §3: `WEDGE_TYPES = 16`, matching the spec's *"Number of directions
    /// for the wedge mask process"* row in the constants table.
    #[test]
    fn wedge_types_constant_matches_spec() {
        assert_eq!(WEDGE_TYPES, 16);
    }

    /// `Default_Wedge_Index_Cdf` is pinned verbatim from §9.4 (p.435) and
    /// satisfies the §8.2.6 well-formedness invariants every other §9.4
    /// table satisfies: the trailing entry of each row is `0` (the §8.3
    /// adaptation counter) and the next-to-last is `1 << 15 == 32768`.
    /// Per the §9.4 note (p.436), indices 0..2, 10..17, and 20..21 in the
    /// first dimension are never used and carry the placeholder uniform
    /// CDF (step 2048 = 32768 / 16).
    #[test]
    fn wedge_index_default_table_pinned() {
        // Shape pins straight off the §9.4 listing.
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF.len(), BLOCK_SIZES);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[0].len(), WEDGE_TYPES + 1);

        // §8.2.6 well-formedness invariants on every row.
        for row in &DEFAULT_WEDGE_INDEX_CDF {
            assert_eq!(row[row.len() - 1], 0);
            assert_eq!(row[row.len() - 2], 1 << 15);
            // Cumulative frequencies strictly increase to 32768.
            for w in row[..row.len() - 1].windows(2) {
                assert!(w[0] < w[1], "cdf must be strictly increasing: {row:?}");
            }
        }

        // §9.4 note: placeholder rows (uniform 16-symbol CDF, step 2048).
        let placeholder: [u16; WEDGE_TYPES + 1] = [
            2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
            28672, 30720, 32768, 0,
        ];
        // Indices where the §9.4 note says the row is never used.
        let placeholder_ix: &[usize] = &[0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21];
        for &i in placeholder_ix {
            assert_eq!(DEFAULT_WEDGE_INDEX_CDF[i], placeholder, "row {i}");
        }

        // Reachable band 3..=9 (BLOCK_8X8..BLOCK_32X32): spec-pinned
        // leading-frequency anchors and last-frequency anchors.
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[3][0], 2438);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[3][14], 31171);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[4][0], 806);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[4][14], 29629);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[5][0], 2779);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[5][14], 31656);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[6][0], 1684);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[6][14], 30877);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[7][0], 1142);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[7][14], 29550);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[8][0], 2742);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[8][14], 31416);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[9][0], 1727);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[9][14], 31012);

        // Reachable band 18..=19 (BLOCK_8X16 / BLOCK_16X8): pin both
        // leading values and one mid-row value each (a transcription
        // typo here would not be caught by either uniform-row or
        // adjacency checks elsewhere).
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[18][0], 154);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[18][7], 23033);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[18][14], 30911);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[19][0], 1135);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[19][7], 21016);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[19][14], 31933);
    }

    /// §8.3.1 `init_non_coeff_cdfs`: the working-copy `wedge_index` field
    /// is seeded verbatim from the §9.4 default.
    #[test]
    fn wedge_index_init_from_defaults() {
        let ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.wedge_index, DEFAULT_WEDGE_INDEX_CDF);
    }

    /// §8.3.2 `wedge_index_cdf` selector returns the right §9.4 row for
    /// every valid `MiSize`, and rejects out-of-range indices with
    /// `None`. Cross-checks the §3 `Wedge_Bits[ MiSize ] > 0` rule: the
    /// placeholder rows match exactly the `Wedge_Bits == 0` positions in
    /// the spec's table (indices 0..2, 10..17, 20..21).
    #[test]
    fn wedge_index_selector_returns_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();
        for (i, want) in DEFAULT_WEDGE_INDEX_CDF.iter().enumerate() {
            let row = ctx.wedge_index_cdf(i).unwrap();
            assert_eq!(row.len(), WEDGE_TYPES + 1);
            assert_eq!(row, want);
        }
        assert!(ctx.wedge_index_cdf(BLOCK_SIZES).is_none());
        assert!(ctx.wedge_index_cdf(BLOCK_SIZES + 7).is_none());

        // §3 `Wedge_Bits[ BLOCK_SIZES ]` table — non-zero only for the
        // reachable rows enumerated above.
        let wedge_bits: [u8; BLOCK_SIZES] = [
            0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0,
        ];
        let placeholder: [u16; WEDGE_TYPES + 1] = [
            2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
            28672, 30720, 32768, 0,
        ];
        for (i, &bits) in wedge_bits.iter().enumerate() {
            if bits == 0 {
                assert_eq!(
                    DEFAULT_WEDGE_INDEX_CDF[i], placeholder,
                    "row {i} (Wedge_Bits == 0) must be the placeholder uniform CDF"
                );
            } else {
                assert_ne!(
                    DEFAULT_WEDGE_INDEX_CDF[i], placeholder,
                    "row {i} (Wedge_Bits > 0) must hold a non-placeholder CDF"
                );
            }
        }
    }

    /// Working-copy independence: mutating the context via the selector
    /// must not touch the §9.4 source table.
    #[test]
    fn wedge_index_working_copy_is_independent() {
        let mut ctx = TileCdfContext::new_from_defaults();
        ctx.wedge_index_cdf(3).unwrap()[0] = 12345;
        ctx.wedge_index_cdf(18).unwrap()[1] = 23456;
        assert_ne!(ctx.wedge_index[3][0], DEFAULT_WEDGE_INDEX_CDF[3][0]);
        assert_ne!(ctx.wedge_index[18][1], DEFAULT_WEDGE_INDEX_CDF[18][1]);
        // Spec source unchanged.
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[3][0], 2438);
        assert_eq!(DEFAULT_WEDGE_INDEX_CDF[18][1], 987);
    }

    /// End-to-end: drive the §8.2 `SymbolDecoder` through a `wedge_index`
    /// default CDF for a `BLOCK_16X16` (`MiSize = 6`, inside the §5.11.28
    /// / §5.11.29 reachable band), confirming the row matches the §9.4
    /// source, the 16-symbol decode lands in `0..WEDGE_TYPES`, and the
    /// working copy adapts.
    #[test]
    fn decode_wedge_index_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.wedge_index;

        // MiSize = 6 (BLOCK_16X16): Wedge_Bits[6] = 4, so wedge_index is
        // coded here per the §5.11.29 `COMPOUND_WEDGE` branch and the
        // §5.11.28 wedge sub-branch.
        let row = tile_ctx.wedge_index_cdf(6).unwrap();
        assert_eq!(row, &DEFAULT_WEDGE_INDEX_CDF[6]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < WEDGE_TYPES,
            "wedge_index must code a symbol in 0..WEDGE_TYPES"
        );

        assert_ne!(
            tile_ctx.wedge_index, before,
            "read_symbol must adapt the working CDF"
        );
    }
}
