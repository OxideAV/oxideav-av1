//! # oxideav-av1
//!
//! **Status:** orphan-rebuild scaffold (post 2026-05-20 audit), clean
//! room rebuild in progress.
//!
//! The decoder/encoder pipeline is not wired up yet. Bitstream
//! parsing has reached:
//!
//!   * **Round 1.** OBU bytestream walker described in §5.3 of the
//!     AV1 Bitstream & Decoding Process Specification — boundaries
//!     in a low-overhead bitstream plus `obu_type` /
//!     `obu_extension_flag` / `obu_has_size_field` / `temporal_id` /
//!     `spatial_id` / `obu_size` fields and a payload slice for each
//!     unit. See [`obu`].
//!
//!   * **Round 2.** Sequence header OBU parse per §5.5
//!     (`sequence_header_obu`, `color_config`, `timing_info`,
//!     `decoder_model_info`, `operating_parameters_info`). Returns a
//!     strongly typed [`sequence_header::SequenceHeader`] descriptor
//!     plus a bit-position so the trailing-bits accounting from
//!     §5.3.1 can plug in cleanly next round. See [`sequence_header`].
//!
//!   * **Round 3.** Leading slice of `uncompressed_header()` per
//!     §5.9.2 — `show_existing_frame` / `frame_to_show_map_idx` /
//!     `display_frame_id` / `frame_type` / `show_frame` /
//!     `showable_frame` / `error_resilient_mode` /
//!     `disable_cdf_update` / `allow_screen_content_tools` /
//!     `force_integer_mv` / `current_frame_id` /
//!     `frame_size_override_flag` / `order_hint` /
//!     `primary_ref_frame` / `refresh_frame_flags`. Composes with
//!     round-2's `SequenceHeader` to drive every conditional read.
//!     See [`frame_header`].
//!
//!   * **Round 4.** Frame-size sub-syntax block per §5.9.5
//!     (`frame_size`) + §5.9.6 (`render_size`) + §5.9.8
//!     (`superres_params`) + §5.9.9 (`compute_image_size`). For
//!     every intra (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frame in the
//!     §5.9.2 syntax tree, [`parse_frame_header`] now drops into
//!     `frame_size()` + `render_size()` after `refresh_frame_flags`
//!     and surfaces the eight-field [`FrameSize`] descriptor
//!     (`frame_width` / `frame_height` / `render_width` /
//!     `render_height` / `superres_denom` / `upscaled_width` /
//!     `mi_cols` / `mi_rows`). The §5.9.7 `frame_size_with_refs()`
//!     `found_ref` shortcut is **not** implemented yet — it reads
//!     `RefUpscaledWidth[]` / `RefFrameHeight[]` /
//!     `RefRenderWidth[]` / `RefRenderHeight[]` from a
//!     reference-frame state table this round does not track —
//!     so inter-frame parsing still stops at `refresh_frame_flags`
//!     with `frame_size = None`. See [`frame_header`].
//!
//!   * **Round 6.** `allow_intrabc` (§5.9.3 path of §5.9.2) +
//!     `tile_info()` (§5.9.15) wired into the streaming
//!     [`parse_frame_header`] walk. For intra frames whose
//!     `allow_screen_content_tools && UpscaledWidth == FrameWidth`
//!     conjunction holds, the parser now consumes the `allow_intrabc`
//!     `f(1)` slot — otherwise the spec's `allow_intrabc = 0`
//!     initialiser stands. After the `frame_size()` / `render_size()`
//!     block (intra path), the parser then walks `tile_info()` per
//!     §5.9.15 and surfaces a typed [`tile_info::TileInfo`]
//!     (`uniform_tile_spacing_flag`, `tile_cols`, `tile_rows`,
//!     `tile_cols_log2`, `tile_rows_log2`, `context_update_tile_id`,
//!     `tile_size_bytes`, `mi_col_starts`, `mi_row_starts`). The
//!     non-uniform-spacing path consumes the `ns(maxWidth)` /
//!     `ns(maxHeight)` `width_in_sbs_minus_1` / `height_in_sbs_minus_1`
//!     fields via the new [`bitreader::BitReader::ns`] primitive
//!     (§4.10.7). Tile-content decode (motion vectors, transform /
//!     quantisation, in-loop filters, film grain) is still out of
//!     scope. See [`tile_info`].
//!
//!   * **Round 5.** Uncompressed-header tail sub-syntaxes — §5.9.10
//!     `read_interpolation_filter()` (returns
//!     [`InterpolationFilter`]), §5.9.11 `loop_filter_params()`
//!     (returns [`LoopFilterParams`] with the `CodedLossless ||
//!     allow_intrabc` short-circuit, the four `loop_filter_level[]`
//!     fields with the `NumPlanes > 1 && (level[0] || level[1])`
//!     gate on the chroma slots, the `f(3)` `loop_filter_sharpness`,
//!     and the `loop_filter_delta_enabled / delta_update /
//!     update_ref_delta[i] / update_mode_delta[i]` per-slot
//!     update walk over `TOTAL_REFS_PER_FRAME = 8` ref-deltas + 2
//!     mode-deltas with `su(7)` signed offsets), and §5.9.12
//!     `quantization_params()` + §5.9.13 `read_delta_q()` (returns
//!     [`QuantizationParams`] with `base_q_idx`, the four
//!     `delta_q_y_dc / delta_q_u_dc / delta_q_u_ac / delta_q_v_dc /
//!     delta_q_v_ac` per-plane offsets, `diff_uv_delta` /
//!     `using_qmatrix` / `qm_y` / `qm_u` / `qm_v`). These remain
//!     available as standalone parser entry points
//!     ([`parse_interpolation_filter`], [`parse_loop_filter_params`],
//!     [`parse_quantization_params`]) for callers that want to
//!     exercise the parsers on a raw byte slice. See
//!     [`uncompressed_header_tail`].
//!
//!   * **Round 7.** §5.9.12 `quantization_params()` and §5.9.14
//!     `segmentation_params()` wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After `tile_info()`
//!     the parser now consumes `quantization_params()` and surfaces a
//!     typed [`QuantizationParams`] on [`FrameHeader::quantization_params`],
//!     then `segmentation_params()` and surfaces a typed
//!     [`SegmentationParams`] on [`FrameHeader::segmentation_params`]
//!     covering `segmentation_enabled`, `segmentation_update_map`,
//!     `segmentation_temporal_update`, `segmentation_update_data`,
//!     the full §5.9.14 `FeatureEnabled[i][j]` /
//!     `FeatureData[i][j]` 8×8 table (`segment_feature_active` /
//!     `segment_feature_data`), and the §5.9.14 trailing
//!     `SegIdPreSkip` / `LastActiveSegId` derivations. The §5.9.14
//!     `primary_ref_frame == PRIMARY_REF_NONE` collapse is honoured
//!     (`update_map = 1`, `temporal_update = 0`, `update_data = 1`,
//!     no bitstream reads for the three update flags). Per-feature
//!     `Segmentation_Feature_Bits` / `Segmentation_Feature_Signed` /
//!     `Segmentation_Feature_Max` Table 5.9.14 tables are exposed as
//!     [`SEGMENTATION_FEATURE_BITS`] / [`SEGMENTATION_FEATURE_SIGNED`]
//!     / [`SEGMENTATION_FEATURE_MAX`]. See
//!     [`uncompressed_header_tail::parse_segmentation_params`].
//!
//!   * **Round 8.** §5.9.17 `delta_q_params()` and §5.9.18
//!     `delta_lf_params()` wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After
//!     `segmentation_params()` the parser consumes `delta_q_params()`
//!     and surfaces a typed [`DeltaQParams`] on
//!     [`FrameHeader::delta_q_params`] (`delta_q_present` — read as
//!     `f(1)` only when `base_q_idx > 0`; `delta_q_res` — `f(2)`,
//!     read only when `delta_q_present == 1`), then `delta_lf_params()`
//!     and surfaces a typed [`DeltaLfParams`] on
//!     [`FrameHeader::delta_lf_params`] (`delta_lf_present` — gated on
//!     `delta_q_present` and suppressed when `allow_intrabc == 1`;
//!     `delta_lf_res` / `delta_lf_multi` — read only when
//!     `delta_lf_present == 1`). Both remain available as standalone
//!     parser entry points
//!     ([`uncompressed_header_tail::parse_delta_q_params`] /
//!     [`uncompressed_header_tail::parse_delta_lf_params`]).
//!   * **Round 9 — §5.9.11 `loop_filter_params()`** wired into the
//!     streaming [`parse_frame_header`] walk (intra path). After
//!     `delta_lf_params()` the parser derives `CodedLossless` from the
//!     §5.9.2 lines that scan `LosslessArray[]` over the per-segment
//!     qindexes (`get_qindex(1, segmentId)` with the §8.7 ignore-delta
//!     branch and the `SEG_LVL_ALT_Q` `Clip3(0, 255, ..)` clamp), then
//!     consumes `loop_filter_params()` and surfaces a typed
//!     [`LoopFilterParams`] on [`FrameHeader::loop_filter_params`]. The
//!     §5.9.11 `CodedLossless || allow_intrabc` short-circuit consumes
//!     no bits and resets the ref-deltas to their §5.9.11 defaults; the
//!     full path reads the four `loop_filter_level[]` slots (chroma pair
//!     gated on `NumPlanes > 1 && (level[0] || level[1])`), the `f(3)`
//!     `loop_filter_sharpness`, and the
//!     `loop_filter_delta_enabled` / `delta_update` per-slot update walk.
//!   * **Round 10 — §5.9.19 `cdef_params()`** wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After
//!     `loop_filter_params()` the parser consumes `cdef_params()` and
//!     surfaces a typed [`CdefParams`] on [`FrameHeader::cdef_params`].
//!     The §5.9.19 `CodedLossless || allow_intrabc || !enable_cdef`
//!     short-circuit consumes no bits and leaves `cdef_bits = 0`,
//!     `CdefDamping = 3`, and all four strength arrays zeroed; the full
//!     path reads `cdef_damping_minus_3` / `cdef_bits` and the
//!     `1 << cdef_bits` `cdef_y_pri` / `cdef_y_sec` (+ `cdef_uv_*` when
//!     `NumPlanes > 1`) strength entries, applying the secondary
//!     `== 3 ⇒ += 1` adjustment. Also available as a standalone parser
//!     entry point ([`parse_cdef_params`]).
//!
//!   * **Round 11 — §5.9.20 `lr_params()`** wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After `cdef_params()`
//!     the parser consumes `lr_params()` and surfaces a typed
//!     [`LrParams`] on [`FrameHeader::lr_params`]. The §5.9.20
//!     `AllLossless || allow_intrabc || !enable_restoration`
//!     short-circuit consumes no bits and leaves every plane
//!     `RESTORE_NONE` with `UsesLr = 0`; the full path reads one
//!     `lr_type` (`f(2)`) per plane (mapped through `Remap_Lr_Type` into
//!     [`FrameRestorationType`]), then — when any plane uses LR —
//!     `lr_unit_shift` (post-incremented for 128×128 superblocks,
//!     otherwise extended by `lr_unit_extra_shift`) and the
//!     4:2:0-chroma-gated `lr_uv_shift`, deriving the three
//!     `LoopRestorationSize[]` entries from `RESTORATION_TILESIZE_MAX`.
//!     Also available as a standalone parser entry point
//!     ([`parse_lr_params`]).
//!
//!   * **Round 12 — §5.9.21 `read_tx_mode()`** wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After `lr_params()` the
//!     parser consumes `read_tx_mode()` and surfaces a typed [`TxMode`]
//!     on [`FrameHeader::tx_mode`]. When `CodedLossless == 1` the §5.9.21
//!     first branch consumes no bits and forces [`TxMode::Only4x4`];
//!     otherwise the `f(1)` `tx_mode_select` slot selects
//!     [`TxMode::TxModeSelect`] (`1`) or [`TxMode::TxModeLargest`] (`0`).
//!     Also available as a standalone parser entry point
//!     ([`parse_tx_mode`]).
//!
//!   * **Round 13 — the §5.9.2 uncompressed-header tail** completes the
//!     intra path. After `read_tx_mode()` the parser consumes
//!     `frame_reference_mode()` (§5.9.23), `skip_mode_params()`
//!     (§5.9.22), the `allow_warped_motion` slot, `reduced_tx_set`
//!     (`f(1)`), `global_motion_params()` (§5.9.24), and
//!     `film_grain_params()` (§5.9.30), surfacing
//!     [`FrameHeader::reference_select`] / `skip_mode_present` /
//!     `allow_warped_motion` / `reduced_tx_set` plus typed
//!     [`GlobalMotionParams`] and [`FilmGrainParams`]. For an intra
//!     frame the first three blocks and `global_motion_params()` all
//!     collapse without reading bits (the §5.9.23 `FrameIsIntra ⇒
//!     reference_select = 0`, the §5.9.22 `skipModeAllowed = 0`, the
//!     §5.9.2 `allow_warped_motion` guard, and the §5.9.24 `FrameIsIntra`
//!     identity short-circuit); `reduced_tx_set` is one bit, and
//!     `film_grain_params()` reads its full §5.9.30 block when the
//!     sequence enables grain and the frame carries `apply_grain == 1`.
//!     `global_motion_params()` exposes the complete §5.9.24/§5.9.25
//!     inter syntax (`read_global_param` + §5.9.26–§5.9.29
//!     `decode_signed_subexp_with_ref` / `decode_subexp` /
//!     `inverse_recenter`) via the standalone
//!     [`parse_global_motion_params`]; `film_grain_params()` is likewise
//!     available via [`parse_film_grain_params`] (taking a
//!     [`FilmGrainContext`]). This is the end of `uncompressed_header()`
//!     for the intra path — [`FrameHeader::bits_consumed`] now reaches
//!     the trailing bits / tile-group boundary.
//!
//!   * **Round 14 — the inter-frame `uncompressed_header()` path.** An
//!     `INTER_FRAME` / `SWITCH_FRAME` header now parses end-to-end. The
//!     §5.9.2 `else` branch reads `frame_refs_short_signaling`, the
//!     explicit `ref_frame_idx[]` (or computes them via §7.8
//!     `set_frame_refs()`), the §5.9.7 `frame_size_with_refs()` /
//!     §5.9.5 `frame_size()` + §5.9.6 `render_size()` size selection,
//!     `allow_high_precision_mv`, §5.9.10 `read_interpolation_filter()`,
//!     `is_motion_mode_switchable`, `use_ref_frame_mvs`, then the shared
//!     `disable_frame_end_update_cdf` + `tile_info()` + quant / segment
//!     / delta / loop-filter / CDEF / LR / `read_tx_mode()` tail, the
//!     inter `frame_reference_mode()` (`reference_select`), §5.9.22
//!     `skip_mode_params()`, `allow_warped_motion`, `reduced_tx_set`,
//!     inter `global_motion_params()`, and `film_grain_params()`. Backed
//!     by a new public [`frame_header::RefInfo`] cross-frame reference
//!     state and surfaced via [`parse_frame_header_with_refs`] /
//!     [`frame_header::InterFrameRefs`] (on
//!     [`FrameHeader::inter_refs`]). Verified byte-exact against the
//!     `i-frame-then-p-64x64` fixture's `idx=1` `FRAME_HEADER` +
//!     `REF_MAP` trace lines.
//!
//!   * **Round 15.** The §8.2 symbol (arithmetic / msac) decoder, as a
//!     standalone [`symbol_decoder::SymbolDecoder`]. Implements §8.2.2
//!     `init_symbol` (the `SymbolValue` / `SymbolRange` / `SymbolMaxBits`
//!     §8.2.4 state), §8.2.6 `read_symbol` (the CDF-adaptive multisymbol
//!     search with `EC_PROB_SHIFT` / `EC_MIN_PROB`, the `prev - cur`
//!     range update, and the seven-step renormalisation that draws new
//!     bits — or §8.2.2 padding zeros once `SymbolMaxBits` is exhausted),
//!     the §8.3 adaptive-rate CDF update, §8.2.3 `read_bool`, §8.2.5
//!     `read_literal` (`L(n)`), `NS(n)` (§4.10.10), the arithmetic-coded
//!     `decode_subexp_bool` (§5.9.28 bool variant), and §8.2.4
//!     `exit_symbol` (trailing-bit accounting + byte-alignment advance,
//!     rejecting the `SymbolMaxBits < -14` conformance violation via
//!     [`Error::SymbolExitUnderflow`]). The default CDF tables (§9.4)
//!     and the §8.3.2 CDF-selection process are out of scope this round —
//!     they land with the tile-content decode that consumes them. See
//!     [`symbol_decoder`].
//!
//!   * **Round 16.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 CDF-selection process for a bounded **intra-frame mode /
//!     partition** syntax group, in a new [`cdf`] module. Transcribes
//!     [`DEFAULT_INTRA_FRAME_Y_MODE_CDF`] (the `intra_frame_y_mode`
//!     5×5×14 table), the five `DEFAULT_PARTITION_W{8,16,32,64,128}_CDF`
//!     tables (the `partition` element), [`DEFAULT_SKIP_CDF`], and
//!     [`DEFAULT_SEGMENT_ID_CDF`] verbatim from §9.4. [`TileCdfContext`]
//!     implements §8.3.1 (`new_from_defaults` copies every `Default_*`
//!     table into a per-tile, mutable `Tile*Cdf` working set that the
//!     §8.2 [`SymbolDecoder`] adapts in place), and the §8.3.2 selection
//!     surfaces a `&mut [u16]` row for each element — `intra_frame_y_mode`
//!     (`[abovemode][leftmode]`), `partition` (array-by-`bsl`, row-by-`ctx`),
//!     `skip` (`[ctx]`), and `segment_id` (`[ctx]`) — feeding
//!     [`SymbolDecoder::read_symbol`] directly. The scalar §8.3.2
//!     context-derivation helpers ([`intra_mode_ctx`] / [`partition_ctx`]
//!     / [`skip_ctx`] / [`segment_id_ctx`]) compute the index from
//!     neighbour inputs the (future) tile walk supplies. The remaining
//!     ~100 §9.4 tables, the `init_coeff_cdfs` coefficient set, and the
//!     other §8.3.2 selections (`split_or_horz` / `split_or_vert` /
//!     `tx_depth` / `txfm_split` / the motion-vector + uv-mode groups)
//!     are a clear followup. See [`cdf`].
//!
//!   * **Round 17.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 CDF-selection process for the **motion-vector component**
//!     syntax group, extending [`cdf`]. Transcribes
//!     [`DEFAULT_MV_JOINT_CDF`], [`DEFAULT_MV_SIGN_CDF`],
//!     [`DEFAULT_MV_CLASS_CDF`], [`DEFAULT_MV_CLASS0_BIT_CDF`],
//!     [`DEFAULT_MV_CLASS0_FR_CDF`], [`DEFAULT_MV_CLASS0_HP_CDF`],
//!     [`DEFAULT_MV_BIT_CDF`], [`DEFAULT_MV_FR_CDF`], and
//!     [`DEFAULT_MV_HP_CDF`] verbatim from §9.4 (the `216*128` /
//!     `136*128` / … fixed-point notation expanded). [`TileCdfContext`]
//!     grows nine `mv_*` working-set fields broadcast per §8.3.1 to
//!     `MV_CONTEXTS = 2` slots (and to the `comp = 0..1` axis where the
//!     source default is per-comp identical), and the §8.3.2 selection
//!     surfaces `&mut [u16]` rows for every MV element: `mv_joint`
//!     (`[MvCtx]`), `mv_sign` / `mv_class` / `mv_class0_bit` /
//!     `mv_class0_hp` / `mv_fr` / `mv_hp` (`[MvCtx][comp]`),
//!     `mv_class0_fr` (`[MvCtx][comp][mv_class0_bit]`), and `mv_bit`
//!     (`[MvCtx][comp][i]`). The §5.11.31 `MvCtx` derivation —
//!     `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0` — is exposed as
//!     the [`mv_ctx`] helper. The remaining ~90 §9.4 tables (y_mode,
//!     uv_mode, angle-delta, tx-size, coefficient, palette, …) and the
//!     other §8.3.2 selections are a mechanical followup against the
//!     same [`TileCdfContext`] shape.
//!
//!   * **Round 18.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the **inter-mode / reference-frame**
//!     syntax group, extending [`cdf`]. Transcribes the 13 remaining
//!     `Default_*_Cdf` tables driving every inter-block mode and
//!     reference syntax: [`DEFAULT_NEW_MV_CDF`], [`DEFAULT_ZERO_MV_CDF`],
//!     [`DEFAULT_REF_MV_CDF`], [`DEFAULT_DRL_MODE_CDF`],
//!     [`DEFAULT_IS_INTER_CDF`], [`DEFAULT_COMP_MODE_CDF`],
//!     [`DEFAULT_SKIP_MODE_CDF`], [`DEFAULT_COMP_REF_CDF`],
//!     [`DEFAULT_COMP_BWD_REF_CDF`], [`DEFAULT_SINGLE_REF_CDF`],
//!     [`DEFAULT_COMPOUND_MODE_CDF`], [`DEFAULT_COMP_REF_TYPE_CDF`],
//!     [`DEFAULT_UNI_COMP_REF_CDF`] verbatim from §9.4, plus the §8.3.2
//!     [`COMPOUND_MODE_CTX_MAP`] lookup table. [`TileCdfContext`] grows
//!     the corresponding `Tile*Cdf` fields, [`TileCdfContext::new_from_defaults`]
//!     seeds them per §8.3.1, and the §8.3.2 selection surfaces
//!     `&mut [u16]` rows for every element: `new_mv_cdf` / `zero_mv_cdf` /
//!     `ref_mv_cdf` / `drl_mode_cdf` / `is_inter_cdf` / `comp_mode_cdf` /
//!     `skip_mode_cdf` / `comp_ref_cdf` / `comp_bwd_ref_cdf` /
//!     `single_ref_cdf` / `compound_mode_cdf` / `comp_ref_type_cdf` /
//!     `uni_comp_ref_cdf`. Scalar §8.3.2 context helpers
//!     [`is_inter_ctx`] (the `(AvailU, AvailL) × (AboveIntra, LeftIntra)`
//!     branch ladder), [`skip_mode_ctx`] (neighbour `SkipModes[]` sum),
//!     [`ref_count_ctx`] (`<` / `==` / `>` three-branch shared by every
//!     `single_ref_p*` / `comp_ref` / `comp_bwdref` / `uni_comp_ref_p*`
//!     paragraph), and [`compound_mode_ctx`] (`Compound_Mode_Ctx_Map`
//!     lookup) compute each `ctx` from the neighbour-summary inputs the
//!     (future) tile walk supplies.
//!
//!   * **Round 20.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the **transform-size** syntax group,
//!     extending [`cdf`]. Transcribes the four per-`maxTxDepth`
//!     `tx_depth` tables ([`DEFAULT_TX_8X8_CDF`], [`DEFAULT_TX_16X16_CDF`],
//!     [`DEFAULT_TX_32X32_CDF`], [`DEFAULT_TX_64X64_CDF`]) and the
//!     binary [`DEFAULT_TXFM_SPLIT_CDF`] verbatim from §9.4.
//!     [`TileCdfContext`] grows the `tx_8x8` / `tx_16x16` /
//!     `tx_32x32` / `tx_64x64` / `txfm_split` fields, all seeded by
//!     [`TileCdfContext::new_from_defaults`] per §8.3.1. The §8.3.2
//!     selection surfaces two `&mut [u16]` accessors —
//!     `tx_depth_cdf(maxTxDepth, ctx)` (the §8.3.2 four-way
//!     `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]` switch keyed by
//!     `Max_Tx_Depth[ MiSize ]`; returns `None` when
//!     `maxTxDepth == 0`) and `txfm_split_cdf(ctx)`. Scalar §8.3.2
//!     context helpers [`tx_depth_ctx`] (the
//!     `(aboveW >= maxTxWidth) + (leftH >= maxTxHeight)` formula) and
//!     [`txfm_split_ctx`] (the
//!     `(txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 +
//!     above + left` formula) compute each `ctx` from scalar inputs
//!     the §5.11.15 / §5.11.16 syntax supplies.
//!
//!   * **Round 21.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the **inter-frame transform-type** syntax
//!     group, extending [`cdf`]. Transcribes the three
//!     `Default_Inter_Tx_Type_Set{1,2,3}_Cdf` tables
//!     ([`DEFAULT_INTER_TX_TYPE_SET1_CDF`],
//!     [`DEFAULT_INTER_TX_TYPE_SET2_CDF`],
//!     [`DEFAULT_INTER_TX_TYPE_SET3_CDF`]) verbatim from §9.4 —
//!     `Set1` for the full 16-symbol set (4x4 / 8x8 inter blocks),
//!     `Set2` for the 16x16 inter set (12 symbols), `Set3` for the
//!     reduced 2-symbol `{ IDTX, DCT_DCT }` set (4x4..32x32 inter
//!     blocks). [`TileCdfContext`] grows the `inter_tx_type_set1` /
//!     `inter_tx_type_set2` / `inter_tx_type_set3` fields, all
//!     seeded by [`TileCdfContext::new_from_defaults`] per §8.3.1.
//!     The §8.3.2 selection surfaces an `&mut [u16]` accessor
//!     `inter_tx_type_cdf(set, tx_size_sqr)` (the §8.3.2 three-way
//!     `TX_SET_INTER_{1,2,3}` switch; returns `None` for
//!     `TX_SET_DCTONLY` per §5.11.47 and for any unreachable
//!     `(set, tx_size_sqr)` combination). The scalar §5.11.48 helper
//!     [`inter_tx_type_set`] computes the `set ∈ { TX_SET_DCTONLY,
//!     TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3 }` from the
//!     `(Tx_Size_Sqr[txSz], Tx_Size_Sqr_Up[txSz], reduced_tx_set)`
//!     tuple supplied by the surrounding §5.11.47 syntax. The intra
//!     counterpart (`Default_Intra_Tx_Type_Set{1,2}_Cdf` with their
//!     extra `intraDir` axis) and the remaining §9.4 tables (y_mode,
//!     uv_mode, angle-delta, coefficient, …) are mechanical
//!     followups against the same [`TileCdfContext`] shape.
//!
//!   * **Round 22.** The §9.4 default CDF table and the §8.3.1 /
//!     §8.3.2 selection for the **inter-frame interpolation-filter**
//!     syntax element, extending [`cdf`]. Transcribes
//!     [`DEFAULT_INTERP_FILTER_CDF`] — `[INTERP_FILTER_CONTEXTS][INTERP_FILTERS + 1]`,
//!     i.e. 16 contexts × 3 cumulative frequencies + adaptation counter
//!     — verbatim from §9.4. New §3 constants `INTERP_FILTERS = 3`
//!     and `INTERP_FILTER_CONTEXTS = 16`. [`TileCdfContext`] grows the
//!     `interp_filter` field, seeded by [`TileCdfContext::new_from_defaults`]
//!     per §8.3.1. The §8.3.2 selection surfaces
//!     `interp_filter_cdf(ctx)`; the scalar [`interp_filter_ctx`]
//!     helper folds the §8.3.2 four-branch
//!     `(above_type, left_type, dir, is_compound)` formula into a
//!     single `0..INTERP_FILTER_CONTEXTS` index (the caller supplies
//!     the already-resolved neighbour-type values per the spec's
//!     `RefFrame[0]` matching predicate). Sentinel
//!     [`INTERP_FILTER_NONE`] (== `INTERP_FILTERS`, mirroring the
//!     spec's literal `3`) marks an unavailable / mismatched neighbour.
//!
//!   * **Round 23.** The §9.4 default CDF table and the §8.3.1 /
//!     §8.3.2 selection for the **motion-mode** syntax element,
//!     extending [`cdf`]. Transcribes [`DEFAULT_MOTION_MODE_CDF`] —
//!     `[BLOCK_SIZES][MOTION_MODES + 1]`, i.e. 22 block-size rows × 3
//!     cumulative frequencies + adaptation counter — verbatim from
//!     §9.4. New §3 constant `MOTION_MODES = 3` (per §6.10.26
//!     `SIMPLE = 0` / `OBMC = 1` / `LOCALWARP = 2`). [`TileCdfContext`]
//!     grows the `motion_mode` field, seeded by
//!     [`TileCdfContext::new_from_defaults`] per §8.3.1. The §8.3.2
//!     selection surfaces `motion_mode_cdf(mi_size)` — a straight
//!     `0..BLOCK_SIZES` index (the spec's selection text reads
//!     "`TileMotionModeCdf[ MiSize ]`"; no neighbour-context
//!     arithmetic).
//!
//! Tile-group / tile-content decode (the per-tile coefficient,
//! motion-vector, and reconstruction passes) remains out of scope, as
//! does the §7.20 reference frame update process that would store a
//! decoded frame back into [`frame_header::RefInfo`] across frames.
//! [`decode_av1`] / [`encode_av1`] continue to return
//! [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod bitreader;
pub mod cdf;
pub mod frame_header;
pub mod obu;
pub mod sequence_header;
pub mod symbol_decoder;
pub mod tile_info;
pub mod uncompressed_header_tail;

pub use cdf::{
    cfl_alpha_u_ctx, cfl_alpha_v_ctx, compound_mode_ctx, inter_tx_type_set, interp_filter_ctx,
    intra_mode_ctx, is_inter_ctx, mv_ctx, palette_color_ctx, palette_uv_mode_ctx,
    palette_y_mode_ctx, partition_ctx, ref_count_ctx, segment_id_ctx, skip_ctx, skip_mode_ctx,
    tx_depth_ctx, txfm_split_ctx, TileCdfContext, BLOCK_SIZES, BWD_REFS, CFL_ALPHABET_SIZE,
    CFL_ALPHA_CONTEXTS, CFL_JOINT_SIGNS, CLASS0_SIZE, COMPOUND_MODES, COMPOUND_MODE_CONTEXTS,
    COMPOUND_MODE_CTX_MAP, COMP_INTER_CONTEXTS, COMP_NEWMV_CTXS, COMP_REF_TYPE_CONTEXTS,
    DEFAULT_CFL_ALPHA_CDF, DEFAULT_CFL_SIGN_CDF, DEFAULT_COMPOUND_MODE_CDF,
    DEFAULT_COMP_BWD_REF_CDF, DEFAULT_COMP_MODE_CDF, DEFAULT_COMP_REF_CDF,
    DEFAULT_COMP_REF_TYPE_CDF, DEFAULT_DRL_MODE_CDF, DEFAULT_FILTER_INTRA_CDF,
    DEFAULT_FILTER_INTRA_MODE_CDF, DEFAULT_INTERP_FILTER_CDF, DEFAULT_INTER_TX_TYPE_SET1_CDF,
    DEFAULT_INTER_TX_TYPE_SET2_CDF, DEFAULT_INTER_TX_TYPE_SET3_CDF, DEFAULT_INTRA_FRAME_Y_MODE_CDF,
    DEFAULT_IS_INTER_CDF, DEFAULT_MOTION_MODE_CDF, DEFAULT_MV_BIT_CDF, DEFAULT_MV_CLASS0_BIT_CDF,
    DEFAULT_MV_CLASS0_FR_CDF, DEFAULT_MV_CLASS0_HP_CDF, DEFAULT_MV_CLASS_CDF, DEFAULT_MV_FR_CDF,
    DEFAULT_MV_HP_CDF, DEFAULT_MV_JOINT_CDF, DEFAULT_MV_SIGN_CDF, DEFAULT_NEW_MV_CDF,
    DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF,
    DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF,
    DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF,
    DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF,
    DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF,
    DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF,
    DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF, DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF,
    DEFAULT_PALETTE_UV_MODE_CDF, DEFAULT_PALETTE_UV_SIZE_CDF, DEFAULT_PALETTE_Y_MODE_CDF,
    DEFAULT_PALETTE_Y_SIZE_CDF, DEFAULT_PARTITION_W128_CDF, DEFAULT_PARTITION_W16_CDF,
    DEFAULT_PARTITION_W32_CDF, DEFAULT_PARTITION_W64_CDF, DEFAULT_PARTITION_W8_CDF,
    DEFAULT_REF_MV_CDF, DEFAULT_SEGMENT_ID_CDF, DEFAULT_SINGLE_REF_CDF, DEFAULT_SKIP_CDF,
    DEFAULT_SKIP_MODE_CDF, DEFAULT_TXFM_SPLIT_CDF, DEFAULT_TX_16X16_CDF, DEFAULT_TX_32X32_CDF,
    DEFAULT_TX_64X64_CDF, DEFAULT_TX_8X8_CDF, DEFAULT_UNI_COMP_REF_CDF, DEFAULT_ZERO_MV_CDF,
    DRL_MODE_CONTEXTS, FWD_REFS, INTERP_FILTERS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_NONE,
    INTER_TX_TYPE_SET1_SIZES, INTER_TX_TYPE_SET3_SIZES, INTRA_FILTER_MODES, INTRA_MODES,
    INTRA_MODE_CONTEXT, INTRA_MODE_CONTEXTS, IS_INTER_CONTEXTS, MAX_TX_DEPTH, MOTION_MODES,
    MV_CLASSES, MV_COMPS, MV_CONTEXTS, MV_INTRABC_CONTEXT, MV_JOINTS, MV_OFFSET_BITS,
    NEW_MV_CONTEXTS, PALETTE_BLOCK_SIZE_CONTEXTS, PALETTE_COLORS, PALETTE_COLOR_CONTEXT,
    PALETTE_COLOR_CONTEXTS, PALETTE_COLOR_HASH_MULTIPLIERS, PALETTE_MAX_COLOR_CONTEXT_HASH,
    PALETTE_NUM_NEIGHBORS, PALETTE_SIZES, PALETTE_UV_MODE_CONTEXTS, PALETTE_Y_MODE_CONTEXTS,
    PARTITION_CONTEXTS, REF_CONTEXTS, REF_MV_CONTEXTS, SEGMENT_ID_CONTEXTS, SINGLE_REFS,
    SKIP_CONTEXTS, SKIP_MODE_CONTEXTS, TXFM_PARTITION_CONTEXTS, TX_SET_DCTONLY, TX_SET_INTER_1,
    TX_SET_INTER_2, TX_SET_INTER_3, TX_SIZES, TX_SIZE_CONTEXTS, TX_TYPES, TX_TYPES_SET2,
    TX_TYPES_SET3, UNIDIR_COMP_REFS, ZERO_MV_CONTEXTS,
};
pub use frame_header::{
    parse_frame_header, parse_frame_header_with_refs, FrameHeader, FrameSize, FrameType,
    InterFrameRefs, RefInfo, NUM_REF_FRAMES, PRIMARY_REF_NONE, SUPERRES_DENOM_BITS,
    SUPERRES_DENOM_MIN, SUPERRES_NUM,
};
pub use obu::{parse_leb128, parse_obu, ObuDescriptor, ObuIter, ObuType};
pub use sequence_header::{
    parse_sequence_header, ColorConfig, DecoderModelInfo, OperatingParametersInfo, OperatingPoint,
    SequenceHeader, TimingInfo,
};
pub use symbol_decoder::SymbolDecoder;
pub use tile_info::{
    parse_tile_info, TileInfo, MAX_TILE_AREA, MAX_TILE_COLS, MAX_TILE_ROWS, MAX_TILE_WIDTH,
};
pub use uncompressed_header_tail::{
    parse_cdef_params, parse_delta_lf_params, parse_delta_q_params, parse_film_grain_params,
    parse_global_motion_params, parse_interpolation_filter, parse_loop_filter_params,
    parse_lr_params, parse_quantization_params, parse_segmentation_params, parse_tx_mode,
    prev_gm_params_default, CdefParams, DeltaLfParams, DeltaQParams, FilmGrainContext,
    FilmGrainParams, FrameRestorationType, GlobalMotionParams, InterpolationFilter,
    LoopFilterParams, LrParams, QuantizationParams, SegmentationParams, TxMode, WarpModelType,
    ALTREF_FRAME, CDEF_MAX_STRENGTHS, GM_ABS_ALPHA_BITS, GM_ABS_TRANS_BITS, GM_ABS_TRANS_ONLY_BITS,
    GM_ALPHA_PREC_BITS, GM_TRANS_ONLY_PREC_BITS, GM_TRANS_PREC_BITS, INTRA_FRAME, LAST_FRAME,
    LOOP_FILTER_MODE_DELTAS_DEFAULT, LOOP_FILTER_REF_DELTAS_DEFAULT, MAX_AR_COEFFS_UV,
    MAX_AR_COEFFS_Y, MAX_LOOP_FILTER, MAX_NUM_CHROMA_POINTS, MAX_NUM_Y_POINTS, MAX_SEGMENTS,
    REFS_PER_FRAME, RESTORATION_TILESIZE_MAX, SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_MAX,
    SEGMENTATION_FEATURE_SIGNED, SEG_LVL_ALT_LF_U, SEG_LVL_ALT_LF_V, SEG_LVL_ALT_LF_Y_H,
    SEG_LVL_ALT_LF_Y_V, SEG_LVL_ALT_Q, SEG_LVL_GLOBALMV, SEG_LVL_MAX, SEG_LVL_REF_FRAME,
    SEG_LVL_SKIP, TOTAL_REFS_PER_FRAME, TX_MODES, WARPEDMODEL_PREC_BITS,
};

/// Crate-local error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A high-level API path is still a scaffold pending the
    /// clean-room rebuild.
    NotImplemented,
    /// The input ended in the middle of an OBU header, extension
    /// header, `leb128()` value, or declared payload.
    UnexpectedEnd,
    /// `obu_forbidden_bit` was set, in violation of §6.2.2.
    ForbiddenBitSet,
    /// The OBU header had `obu_has_size_field == 0`; the walker only
    /// accepts the §5.2 low-overhead format with explicit sizes.
    MissingSizeField,
    /// A `leb128()` value exceeded `(1 << 32) - 1`, the §4.10.5
    /// bitstream-conformance cap.
    Leb128Overflow,
    /// A `leb128()` encoding consumed more than 8 bytes — §4.10.5
    /// requires the MSB of the 8th byte to be 0.
    Leb128TooLong,
    /// An `obu_size` value did not fit in `usize` on this target.
    SizeOverflow,
    /// `seq_profile` was greater than 2 — values 3..=7 are reserved
    /// per §6.4.1.
    ReservedProfile(u8),
    /// `reduced_still_picture_header == 1` but `still_picture == 0`,
    /// in violation of the §6.4.1 conformance requirement.
    ReducedStillRequiresStill,
    /// `idLen` (= `additional_frame_id_length_minus_1 +
    /// `delta_frame_id_length_minus_2 + 3`) exceeded the §6.8.2
    /// requirement that the bit width of `display_frame_id` /
    /// `current_frame_id` must not exceed 16.
    InvalidIdLen,
    /// The frame-header parser hit a `temporal_point_info()` call
    /// site (§5.9.31) — i.e. `decoder_model_info_present_flag &&
    /// !equal_picture_interval`. Decoder-model frame timing isn't
    /// implemented yet; every fixture in this round's corpus parses
    /// without ever triggering this path.
    TemporalPointInfoUnsupported,
    /// Retained for API stability. The §5.9.2 `if (!FrameIsIntra ||
    /// refresh_frame_flags != allFrames) { if (error_resilient_mode &&
    /// enable_order_hint) { ... } }` ref_order_hint walk is now parsed
    /// (the bits are consumed; the conformance-only `RefValid[i] = 0`
    /// invalidation against the session's `RefOrderHint[]` has no effect
    /// on the parse), so the inter-frame header path no longer returns
    /// this variant. It is kept to avoid a breaking enum change.
    RefOrderHintWalkUnsupported,
    /// `exit_symbol()` was invoked while `SymbolMaxBits` was strictly
    /// less than `-14`, violating the §8.2.4 bitstream-conformance
    /// requirement that `SymbolMaxBits >= -14` at exit.
    SymbolExitUnderflow,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "oxideav-av1: orphan-rebuild scaffold — no decoder/encoder wired up"
            ),
            Self::UnexpectedEnd => write!(f, "oxideav-av1: unexpected end of OBU bytestream"),
            Self::ForbiddenBitSet => {
                write!(f, "oxideav-av1: obu_forbidden_bit was set (§6.2.2)")
            }
            Self::MissingSizeField => write!(
                f,
                "oxideav-av1: obu_has_size_field == 0; only the §5.2 low-overhead format is supported"
            ),
            Self::Leb128Overflow => {
                write!(f, "oxideav-av1: leb128 value exceeded the §4.10.5 cap")
            }
            Self::Leb128TooLong => write!(
                f,
                "oxideav-av1: leb128 encoding used more than 8 bytes (§4.10.5)"
            ),
            Self::SizeOverflow => {
                write!(f, "oxideav-av1: obu_size did not fit in usize on this target")
            }
            Self::ReservedProfile(p) => write!(
                f,
                "oxideav-av1: seq_profile {p} is reserved (only 0..=2 are conformant, §6.4.1)"
            ),
            Self::ReducedStillRequiresStill => write!(
                f,
                "oxideav-av1: reduced_still_picture_header == 1 requires still_picture == 1 (§6.4.1)"
            ),
            Self::InvalidIdLen => write!(
                f,
                "oxideav-av1: idLen (delta_frame_id_length_minus_2 + additional_frame_id_length_minus_1 + 3) exceeded 16 (§6.8.2)"
            ),
            Self::TemporalPointInfoUnsupported => write!(
                f,
                "oxideav-av1: temporal_point_info() / decoder-model framing not implemented yet (§5.9.31)"
            ),
            Self::RefOrderHintWalkUnsupported => write!(
                f,
                "oxideav-av1: ref_order_hint walk in §5.9.2 needs RefOrderHint[] state (not yet tracked)"
            ),
            Self::SymbolExitUnderflow => write!(
                f,
                "oxideav-av1: exit_symbol() with SymbolMaxBits < -14 (§8.2.4 conformance)"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Decode an AV1 elementary stream.
///
/// Still a stub: this round only added the OBU bytestream walker.
pub fn decode_av1(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Encode YUV data into an AV1 elementary stream.
pub fn encode_av1(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the clean-room scaffold does not yet
/// register a working decoder or encoder.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("av1", register);
