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
//!   * **Round 24.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the **compound-prediction** syntax
//!     elements, extending [`cdf`]. Transcribes
//!     [`DEFAULT_COMP_GROUP_IDX_CDF`] (`[COMP_GROUP_IDX_CONTEXTS][3]`),
//!     [`DEFAULT_COMPOUND_IDX_CDF`] (`[COMPOUND_IDX_CONTEXTS][3]`) and
//!     [`DEFAULT_COMPOUND_TYPE_CDF`]
//!     (`[BLOCK_SIZES][COMPOUND_TYPES + 1]`) verbatim from §9.4. New §3
//!     constants `COMPOUND_TYPES = 2`, `COMP_GROUP_IDX_CONTEXTS = 6`,
//!     `COMPOUND_IDX_CONTEXTS = 6`. [`TileCdfContext`] grows the
//!     `comp_group_idx` / `compound_idx` / `compound_type` fields,
//!     seeded by [`TileCdfContext::new_from_defaults`] per §8.3.1. The
//!     §8.3.2 selection surfaces `comp_group_idx_cdf(ctx)` /
//!     `compound_idx_cdf(ctx)` (binary, precomputed-`ctx` index) and
//!     `compound_type_cdf(mi_size)` (a straight `0..BLOCK_SIZES` index;
//!     the spec's selection text reads "`TileCompoundTypeCdf[ MiSize
//!     ]`").
//!
//!   * **Round 134.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the inter-frame **intra-mode** syntax
//!     elements, extending [`cdf`]. Transcribes [`DEFAULT_Y_MODE_CDF`]
//!     (`[BLOCK_SIZE_GROUPS][INTRA_MODES + 1]`),
//!     [`DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF`]
//!     (`[INTRA_MODES][UV_INTRA_MODES_CFL_NOT_ALLOWED + 1]`) and
//!     [`DEFAULT_UV_MODE_CFL_ALLOWED_CDF`]
//!     (`[INTRA_MODES][UV_INTRA_MODES_CFL_ALLOWED + 1]`) verbatim from
//!     §9.4. New §3 constants `BLOCK_SIZE_GROUPS = 4`,
//!     `UV_INTRA_MODES_CFL_NOT_ALLOWED = 13`,
//!     `UV_INTRA_MODES_CFL_ALLOWED = 14` plus the §8.3.2
//!     [`SIZE_GROUP`] table. [`TileCdfContext`] grows the `y_mode` /
//!     `uv_mode_cfl_not_allowed` / `uv_mode_cfl_allowed` fields, seeded
//!     by [`TileCdfContext::new_from_defaults`] per §8.3.1. The §8.3.2
//!     selection surfaces `y_mode_cdf(ctx)` (with [`size_group`]
//!     performing the `ctx = Size_Group[ MiSize ]` mapping) and
//!     `uv_mode_cdf(cfl_allowed, y_mode)` (picking the cfl-allowed /
//!     cfl-not-allowed variant by the resolved flag, then indexing by
//!     `YMode`).
//!
//!   * **Round 135.** The §9.4 default CDF table and the §8.3.1 /
//!     §8.3.2 selection for the **angle-delta** syntax elements
//!     (`angle_delta_y` / `angle_delta_uv`), extending [`cdf`].
//!     Transcribes [`DEFAULT_ANGLE_DELTA_CDF`]
//!     (`[DIRECTIONAL_MODES][(2 * MAX_ANGLE_DELTA + 1) + 1]`) verbatim
//!     from §9.4. New §3 constants `DIRECTIONAL_MODES = 8`,
//!     `MAX_ANGLE_DELTA = 3` and the directional-mode base `V_PRED = 1`.
//!     [`TileCdfContext`] grows the `angle_delta` field, seeded by
//!     [`TileCdfContext::new_from_defaults`] per §8.3.1. The §8.3.2
//!     selection surfaces `angle_delta_cdf(mode)`, which rebases a
//!     directional `YMode` / `UVMode` onto `0..DIRECTIONAL_MODES` via
//!     `mode - V_PRED` (the `TileAngleDeltaCdf[ YMode - V_PRED ]` /
//!     `TileAngleDeltaCdf[ UVMode - V_PRED ]` selection).
//!   * **Round 136.** The §9.4 default CDF tables and the §8.3.1
//!     `init_coeff_cdfs` / §8.3.2 selection for the **coefficient-token
//!     entry sub-group** — the gateway to tile-content decode: the
//!     transform-block skip flag ([`DEFAULT_TXB_SKIP_CDF`]), the
//!     end-of-block position class
//!     ([`DEFAULT_EOB_PT_16_CDF`] … [`DEFAULT_EOB_PT_1024_CDF`]), the
//!     EOB extra-bit ([`DEFAULT_EOB_EXTRA_CDF`]), and the DC sign
//!     ([`DEFAULT_DC_SIGN_CDF`]), all transcribed verbatim from §9.4.
//!     New §3 constants `PLANE_TYPES = 2`, `COEFF_CDF_Q_CTXS = 4`,
//!     `TXB_SKIP_CONTEXTS = 13`, `EOB_COEF_CONTEXTS = 9`,
//!     `DC_SIGN_CONTEXTS = 3`. Unlike the non-coeff CDFs, these are
//!     reset by the separate [`TileCdfContext::init_coeff_cdfs`]
//!     (`base_q_idx` → `idx` via [`coeff_cdf_q_ctx`]) so the working
//!     copy drops the `COEFF_CDF_Q_CTXS` axis, selecting
//!     `Default_*_Cdf[ idx ]`. The coeff_base / coeff_base_eob /
//!     coeff_br braid is deferred to a later round.
//!
//!   * **Round 137.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the **intra-frame transform-type** syntax
//!     group, completing the §6.10.19 transform-set coverage started
//!     in round 21. Transcribes [`DEFAULT_INTRA_TX_TYPE_SET1_CDF`]
//!     (`[INTRA_TX_TYPE_SET1_SIZES][INTRA_MODES][TX_TYPES_INTRA_SET1 + 1]`,
//!     i.e. `[2][13][8]` per the §9.4 listing) and
//!     [`DEFAULT_INTRA_TX_TYPE_SET2_CDF`]
//!     (`[INTRA_TX_TYPE_SET2_SIZES][INTRA_MODES][TX_TYPES_INTRA_SET2 + 1]`,
//!     i.e. `[3][13][6]`) verbatim from §9.4 — `Set1` for the full
//!     7-symbol intra set (4x4 / 8x8 intra blocks), `Set2` for the
//!     reduced 5-symbol set (4x4 / 8x8 / 16x16 intra blocks). Both
//!     tables carry an extra `intraDir` axis (`INTRA_MODES = 13`) on
//!     top of the `Tx_Size_Sqr` axis already seen in the inter
//!     counterparts. New §3 constants `TX_SET_INTRA_1 = 1`,
//!     `TX_SET_INTRA_2 = 2`, `TX_TYPES_INTRA_SET1 = 7`,
//!     `TX_TYPES_INTRA_SET2 = 5`, `INTRA_TX_TYPE_SET1_SIZES = 2`,
//!     `INTRA_TX_TYPE_SET2_SIZES = 3`, and the §8.3.2
//!     [`FILTER_INTRA_MODE_TO_INTRA_DIR`] table. [`TileCdfContext`]
//!     grows the `intra_tx_type_set1` / `intra_tx_type_set2` fields,
//!     seeded by [`TileCdfContext::new_from_defaults`] per §8.3.1.
//!     The §8.3.2 selection surfaces an `&mut [u16]` accessor
//!     `intra_tx_type_cdf(set, tx_size_sqr, intra_dir)` (the §8.3.2
//!     two-way `TX_SET_INTRA_{1,2}` switch over the §8.3.2
//!     `intraDir` axis; returns `None` for `TX_SET_DCTONLY` per
//!     §5.11.47 and for any unreachable `(set, tx_size_sqr,
//!     intra_dir)` combination). The scalar §5.11.48 helper
//!     [`intra_tx_type_set`] computes the `set ∈ { TX_SET_DCTONLY,
//!     TX_SET_INTRA_1, TX_SET_INTRA_2 }` from the `(Tx_Size_Sqr[txSz],
//!     Tx_Size_Sqr_Up[txSz], reduced_tx_set)` tuple supplied by the
//!     surrounding §5.11.47 syntax (the intra branch differs from the
//!     inter one in two places: `txSzSqrUp == TX_32X32` itself routes
//!     to `TX_SET_DCTONLY` rather than `TX_SET_INTER_3`, and the
//!     `txSzSqr == TX_16X16` branch routes to `TX_SET_INTRA_2`
//!     rather than `TX_SET_INTER_2`). The scalar §8.3.2 helper
//!     [`intra_dir`] derives the `intraDir` axis from the
//!     `use_filter_intra` flag plus the `YMode` / `filter_intra_mode`
//!     pair.
//!
//!   * **Round 138.** The §9.4 default CDF table and the §8.3.1
//!     `init_coeff_cdfs` / §8.3.2 selection for the first member of the
//!     `coeff_base` / `coeff_base_eob` / `coeff_br` braid:
//!     [`DEFAULT_COEFF_BASE_EOB_CDF`]
//!     (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS_EOB][4]`),
//!     transcribed verbatim from §9.4. `coeff_base_eob` codes the base
//!     level of the last non-zero coefficient (the base level is
//!     `coeff_base_eob + 1`; only 1, 2, or 3 are coded). New §3
//!     constant `SIG_COEF_CONTEXTS_EOB = 4`. [`TileCdfContext`] grows
//!     the `coeff_base_eob` field, seeded by
//!     [`TileCdfContext::new_from_defaults`] from the `idx == 0`
//!     slice and re-selected per `base_q_idx` by
//!     [`TileCdfContext::init_coeff_cdfs`]. The §8.3.2 selection
//!     surfaces `coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx)`, the
//!     three-way `TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]`
//!     lookup; the §8.3.2 ctx derivation
//!     (`get_coeff_base_ctx() - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`)
//!     belongs to the not-yet-implemented tile-content walk and is
//!     deferred. The remaining two tables of the braid
//!     (`Default_Coeff_Base_Cdf` and `Default_Coeff_Br_Cdf`) are
//!     deferred to later rounds.
//!
//!   * **Round 139.** The §9.4 default CDF table and the §8.3.1
//!     `init_coeff_cdfs` / §8.3.2 selection for the second member of
//!     the `coeff_base` / `coeff_base_eob` / `coeff_br` braid:
//!     [`DEFAULT_COEFF_BASE_CDF`]
//!     (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS][5]`,
//!     1680 5-entry rows; declared `static` so `clippy::large_const_arrays`
//!     does not flag the per-use copy hazard), transcribed verbatim
//!     from §9.4. `coeff_base` codes the base level of each non-EOB
//!     coefficient — a 4-symbol alphabet (`0..3`). New §3 constant
//!     `SIG_COEF_CONTEXTS = 42`. [`TileCdfContext`] grows the
//!     `coeff_base` field, seeded by
//!     [`TileCdfContext::new_from_defaults`] from the `idx == 0`
//!     slice and re-selected per `base_q_idx` by
//!     [`TileCdfContext::init_coeff_cdfs`]. The §8.3.2 selection
//!     surfaces `coeff_base_cdf(tx_sz_ctx, ptype, ctx)`, the
//!     three-way `TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]`
//!     lookup; the `get_coeff_base_ctx()` derivation itself belongs
//!     to the not-yet-implemented tile-content walk and is deferred.
//!     The largest `(TX_SIZE = TX_64X64, ptype = chroma)` slice is
//!     a flat `{8192, 16384, 24576, 32768, 0}` placeholder across
//!     every q-context and ctx value (an unreachable-chroma sentinel
//!     mirroring the r138 (tx4, pt1) placeholder pattern). The last
//!     remaining table of the braid (`Default_Coeff_Br_Cdf`) is
//!     deferred to a later round.
//!
//!   * **Round 140.** The §9.4 default CDF table and the §8.3.1
//!     `init_coeff_cdfs` / §8.3.2 selection for the LAST member of
//!     the `coeff_base` / `coeff_base_eob` / `coeff_br` braid:
//!     [`DEFAULT_COEFF_BR_CDF`]
//!     (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][LEVEL_CONTEXTS][BR_CDF_SIZE + 1]`,
//!     840 5-entry rows; declared `static` so `clippy::large_const_arrays`
//!     does not flag the per-use copy hazard), transcribed verbatim
//!     from §9.4. With this table all three coefficient-CDF braid
//!     members are landed. `coeff_br` codes the per-coefficient
//!     base-range increment used to push a level above
//!     `NUM_BASE_LEVELS`: each read codes a value in
//!     `0..BR_CDF_SIZE = 4`, and §5.11.39 stacks
//!     `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)` such reads per
//!     coefficient. New §3 constants `LEVEL_CONTEXTS = 21` and
//!     `BR_CDF_SIZE = 4`. [`TileCdfContext`] grows the `coeff_br`
//!     field, seeded by [`TileCdfContext::new_from_defaults`] from
//!     the `idx == 0` slice and re-selected per `base_q_idx` by
//!     [`TileCdfContext::init_coeff_cdfs`]. The §8.3.2 selection
//!     surfaces `coeff_br_cdf(tx_sz_ctx, ptype, ctx)`, implementing
//!     the spec selector
//!     `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]`
//!     with the `TX_32X32 = 3` clamp built in (so any `txSzCtx` is
//!     accepted; only `ptype` / `ctx` are bounds-checked); the
//!     `get_br_ctx()` derivation itself belongs to the
//!     not-yet-implemented tile-content walk and is deferred. The
//!     largest `(TX_SIZE = TX_64X64, ptype = chroma)` slice is again
//!     a flat `{8192, 16384, 24576, 32768, 0}` placeholder, mirroring
//!     the r138 / r139 pattern. The next gate is the §8.3.2
//!     `get_coeff_base_ctx()` / `get_br_ctx()` neighbour-derivation
//!     helpers (deferred to a different round — they need
//!     tile-content walker state).
//!
//!   * **Round 141.** The §8.3.2 neighbour-derivation helpers that
//!     close out the coefficient-CDF braid: [`get_coeff_base_ctx`]
//!     and [`get_br_ctx`] (plus the [`get_coeff_base_eob_ctx`]
//!     wrapper that subtracts `SIG_COEF_CONTEXTS` and adds
//!     `SIG_COEF_CONTEXTS_EOB` to land on the EOB-CDF context
//!     range). Each consumes a coefficient-magnitude array `Quant[]`
//!     (laid out row-major with stride `Tx_Width[ adjTxSz ]`) plus
//!     the position-in-scan + tx-class + tx-size and returns the
//!     `ctx` index consumed by the matching `coeff_base_eob_cdf` /
//!     `coeff_base_cdf` / `coeff_br_cdf` selector. [`get_tx_class`]
//!     performs the §8.3.2 `TxType -> TxClass` reduction
//!     (`V_DCT` / `V_ADST` / `V_FLIPADST` -> [`TX_CLASS_VERT`],
//!     `H_DCT` / `H_ADST` / `H_FLIPADST` -> [`TX_CLASS_HORIZ`],
//!     everything else -> [`TX_CLASS_2D`]). New §3 constants
//!     [`TX_SIZES_ALL`] (`= 19`), [`SIG_COEF_CONTEXTS_2D`] (`= 26`),
//!     [`SIG_REF_DIFF_OFFSET_NUM`] (`= 5`), [`NUM_BASE_LEVELS`]
//!     (`= 2`), [`COEFF_BASE_RANGE`] (`= 12`), [`TX_CLASS_2D`],
//!     [`TX_CLASS_HORIZ`], [`TX_CLASS_VERT`]. New §-Additional-
//!     tables transcriptions [`TX_WIDTH`], [`TX_HEIGHT`],
//!     [`TX_WIDTH_LOG2`], [`ADJUSTED_TX_SIZE`],
//!     [`SIG_REF_DIFF_OFFSET`], [`MAG_REF_OFFSET_WITH_TX_CLASS`],
//!     [`COEFF_BASE_CTX_OFFSET`], [`COEFF_BASE_POS_CTX_OFFSET`].
//!     Caller still owns the tile-content walk that supplies
//!     `Quant[]` itself plus the `compute_tx_type()` derivation that
//!     feeds [`get_tx_class`]; this round delivers the per-coefficient
//!     `ctx` plumbing those callers will consume.
//!
//!   * **Round 142.** The §5.11.40 `compute_tx_type()` derivation —
//!     [`compute_tx_type`], the per-plane / per-block transform-type
//!     lookup the tile-content walker reads before kicking off
//!     coefficient decoding. Implements the full spec function:
//!     `Lossless || Tx_Size_Sqr_Up[ txSz ] > TX_32X32` short-circuits
//!     to `DCT_DCT`; `plane == 0` returns the `TxTypes[ blockY ][
//!     blockX ]` luma cache entry; `is_inter` chroma reads the cache
//!     at `(Max(MiRow, blockY << subsampling_y),
//!     Max(MiCol, blockX << subsampling_x))` then runs the
//!     [`is_tx_type_in_set`] admission filter; intra chroma reads
//!     [`MODE_TO_TXFM`]`[UVMode]` then runs the same filter. The
//!     §5.11.40 `is_tx_type_in_set(txSet, txType)` predicate is a
//!     direct read of [`TX_TYPE_IN_SET_INTER`] /
//!     [`TX_TYPE_IN_SET_INTRA`]. The caller supplies the §5.11.40
//!     `txSet` (i.e. the already-resolved [`inter_tx_type_set`] /
//!     [`intra_tx_type_set`] result) and a closure over
//!     `TxTypes[y][x]` so the helper does not bake in a particular
//!     storage shape. New §6.10.16 ordinal constants [`TX_4X4`] /
//!     [`TX_8X8`] / [`TX_16X16`] / [`TX_32X32`] / [`TX_64X64`]
//!     replace the local `const TX_*` shadows the existing
//!     [`inter_tx_type_set`] / [`intra_tx_type_set`] helpers used.
//!     New §6.10.19 transform-type ordinals [`DCT_DCT`] through
//!     [`H_FLIPADST`] (16 entries) name the previously-numeric
//!     entries. New §-Additional-tables transcription
//!     [`TX_SIZE_SQR_UP`] (`Tx_Size_Sqr_Up[ TX_SIZES_ALL ]`,
//!     `t -> Max(w, h)-sided square`). New §5.11.40 tables
//!     [`MODE_TO_TXFM`] (chroma-mode -> default transform-type),
//!     [`TX_TYPE_IN_SET_INTRA`] (`TX_SET_TYPES_INTRA = 3` rows of
//!     `TX_TYPES = 16` admission flags), [`TX_TYPE_IN_SET_INTER`]
//!     (`TX_SET_TYPES_INTER = 4` rows). The derivation is pure / msac-
//!     independent — the tile walker plugs it in once `MiRow` /
//!     `MiCol` / `subsampling_x` / `subsampling_y` / `UVMode` /
//!     `TxTypes` state lands. 278 -> 288 tests, zero `#[ignore]`.
//!
//!   * **Round 143.** The §9.4 default CDF tables and the §8.3.1 /
//!     §8.3.2 selection for the **inter-intra** syntax group
//!     (`interintra`, `interintra_mode`, `wedge_interintra` — the
//!     §5.11.28 `read_interintra_mode` triple). Transcribes
//!     [`DEFAULT_INTER_INTRA_CDF`] (`[BLOCK_SIZE_GROUPS - 1][3]`),
//!     [`DEFAULT_INTER_INTRA_MODE_CDF`]
//!     (`[BLOCK_SIZE_GROUPS - 1][INTERINTRA_MODES + 1]`) and
//!     [`DEFAULT_WEDGE_INTER_INTRA_CDF`] (`[BLOCK_SIZES][3]`) verbatim
//!     from §9.4. New §3 constant [`INTERINTRA_MODES`] (`= 4`, per
//!     §6.10.27 `II_DC_PRED` / `II_V_PRED` / `II_H_PRED` /
//!     `II_SMOOTH_PRED`). [`TileCdfContext`] grows the `inter_intra` /
//!     `inter_intra_mode` / `wedge_inter_intra` fields, seeded by
//!     [`TileCdfContext::new_from_defaults`] per §8.3.1. The §8.3.2
//!     selection surfaces `inter_intra_cdf(ctx)` /
//!     `inter_intra_mode_cdf(ctx)` (the scalar [`interintra_ctx`]
//!     helper folds the spec's `ctx = Size_Group[ MiSize ] - 1` into a
//!     single `0..(BLOCK_SIZE_GROUPS - 1)` index, returning `None` for
//!     the `Size_Group[ MiSize ] == 0` rows that the §5.11.28 syntax
//!     gate excludes) and `wedge_inter_intra_cdf(mi_size)` (a straight
//!     `0..BLOCK_SIZES` index). The wedge table's outer dimension is
//!     transcribed full-width per the §9.4 listing; per its note only
//!     indices `3..=9` (the `BLOCK_8X8`..`BLOCK_32X32` band) are
//!     reachable. 288 -> 296 tests, zero `#[ignore]`.
//!
//!   * **Round 144.** The §9.4 default CDF table and the §8.3.1 /
//!     §8.3.2 selection for the **wedge-index** syntax element —
//!     `wedge_index`, read by both §5.11.28 `read_interintra_mode` (the
//!     inter-intra wedge sub-branch, when `wedge_interintra == 1`) and
//!     §5.11.29 `read_compound_type` (the inter-inter `COMPOUND_WEDGE`
//!     branch). Transcribes [`DEFAULT_WEDGE_INDEX_CDF`]
//!     (`[BLOCK_SIZES][WEDGE_TYPES + 1]`) verbatim from §9.4 (p.435).
//!     New §3 constant [`WEDGE_TYPES`] (`= 16`, the spec text reads
//!     *"Number of directions for the wedge mask process"*).
//!     [`TileCdfContext`] grows the `wedge_index` field, seeded by
//!     [`TileCdfContext::new_from_defaults`] per §8.3.1. The §8.3.2
//!     selection `wedge_index_cdf(mi_size)` is a straight
//!     `TileWedgeIndexCdf[ MiSize ]` index. The table's outer dimension
//!     is transcribed full-width per the §9.4 listing; per its note
//!     (p.436) indices 0..2, 10..17, and 20..21 are never used in the
//!     first dimension (matching the §3 `Wedge_Bits[ MiSize ] == 0`
//!     rows) and carry the placeholder uniform CDF `{ 2048, 4096, …,
//!     30720, 32768, 0 }` (step `32768 / WEDGE_TYPES`). 296 -> 302
//!     tests, zero `#[ignore]`.
//!
//!   * **Round 145.** The §8.3.2 `split_or_horz` / `split_or_vert`
//!     cdf-derivation helpers — [`split_or_horz_cdf`] and
//!     [`split_or_vert_cdf`] — that build a 2-symbol binary cdf out of
//!     the already-selected `partition` cdf (the spec's `partitionCdf`)
//!     per p.362. Each helper folds the §9.4 partition probabilities of
//!     the "splittable plus orthogonal-axis" symbols into a single
//!     `psum`, then emits `cdf[0] = (1 << 15) - psum`,
//!     `cdf[1] = 1 << 15`, `cdf[2] = 0`; per the §8.3.2 note the
//!     probability of the disallowed orthogonal partition gets folded
//!     into the split probability, so `split_or_horz` cannot return a
//!     `PARTITION_VERT` and `split_or_vert` cannot return a
//!     `PARTITION_HORZ`. The `b_size != BLOCK_128X128` guard drops the
//!     `PARTITION_*_4` term that the §9.4 `Default_Partition_W128_Cdf`
//!     row has no entry for. New §6.10.4 / §3 ordinal constants
//!     [`PARTITION_NONE`] / [`PARTITION_HORZ`] / [`PARTITION_VERT`] /
//!     [`PARTITION_SPLIT`] / [`PARTITION_HORZ_A`] / [`PARTITION_HORZ_B`]
//!     / [`PARTITION_VERT_A`] / [`PARTITION_VERT_B`] /
//!     [`PARTITION_HORZ_4`] / [`PARTITION_VERT_4`] +
//!     [`EXT_PARTITION_TYPES`] (`= 10`) + the block-size ordinal
//!     [`BLOCK_128X128`] (`= 15`) replace the literal indices the §8.3.2
//!     formulas use. 302 -> 312 tests, zero `#[ignore]`.
//!
//!   * **Round 146.** The §5.11.50 `get_palette_color_context`
//!     derivation — the function the §5.11.49 diagonal walk reads at
//!     each `palette_color_idx_*` position to produce the
//!     `ColorOrder[ PALETTE_COLORS ]` permutation +
//!     `ColorContextHash` that flow back through [`palette_color_ctx`]
//!     into the §8.3.2 cdf selector. Surface:
//!     [`palette_color_context_from_neighbors`] (pure-scoring core
//!     taking the three optional neighbour palette indices) and
//!     [`get_palette_color_context`] (spec-faithful 2-D entry that
//!     applies the §5.11.50 `r > 0` / `c > 0` boundary guards). The
//!     [`PaletteColorContext`] struct packages both outputs. The
//!     partial selection sort is the §5.11.50 three-iteration loop
//!     that promotes the top-scoring neighbours to the head of
//!     `ColorOrder` while preserving the runners-up's ascending
//!     order; the hash is the `Palette_Color_Hash_Multipliers`-
//!     weighted sum of the top three sorted scores. 312 -> 323
//!     tests, zero `#[ignore]`.
//!
//!   * **Round 147.** The §5.11.49 `palette_tokens( )` per-plane
//!     diagonal walker — the §5.11.49 caller-facing entry point that
//!     drives the §5.11.50 colour-context derivation across one of the
//!     planes' anti-diagonal walk, decodes one
//!     `palette_color_idx_{y,uv}` per `(i - j, j)` against the cdf
//!     row picked by [`palette_color_ctx`], remaps through
//!     `ColorOrder[idx]`, and fills the on-screen border by
//!     replicating the last on-screen column / row. Surface:
//!     [`palette_tokens_plane`] (driving the [`SymbolDecoder`] +
//!     [`TileCdfContext`]) and the [`PalettePlane`] selector. Two
//!     new [`Error`] variants surface caller-bug preconditions
//!     ([`Error::InvalidPaletteWalkArgs`]) and the §5.11.50
//!     unreachable hash slots ([`Error::PaletteColorContextUnmapped`]).
//!     323 -> 334 tests, zero `#[ignore]`.
//!
//!   * **Round 148.** The §9.3 block-size conversion tables
//!     (av1-spec p.400–401) — four `BLOCK_SIZES`-indexed lookup
//!     tables that turn a `MiSize` into block dimensions:
//!     [`MI_WIDTH_LOG2`], [`MI_HEIGHT_LOG2`], [`NUM_4X4_BLOCKS_WIDE`],
//!     [`NUM_4X4_BLOCKS_HIGH`], plus the §3 constants
//!     [`MI_SIZE`] (`4`) and [`MI_SIZE_LOG2`] (`2`). Surface: the
//!     four tables plus the six `MiSize`-keyed `const fn` accessors
//!     [`block_width`], [`block_height`], [`num_4x4_blocks_wide`],
//!     [`num_4x4_blocks_high`], [`mi_width_log2`],
//!     [`mi_height_log2`]. The `Block_Width[ x ] = 4 *
//!     Num_4x4_Blocks_Wide[ x ]` spec identity is encoded as the
//!     `NUM_4X4_BLOCKS_WIDE[ x ] << MI_SIZE_LOG2` shift so the
//!     identity is not duplicated as a numeric table. These feed the
//!     §5.11.49 [`palette_tokens_plane`] caller staged in r147 and
//!     unblock the wider §5.x reconstruction call sites that the
//!     parser will surface once `read_block` is wired. 334 -> 344
//!     tests, zero `#[ignore]`.
//!
//!   * **Round 149.** The §5.11.49 caller-side argument derivation
//!     (av1-spec p.101-102) — [`palette_tokens_args`] computes the
//!     four [`PaletteTokensArgs`] dimensions (`blockWidth`,
//!     `blockHeight`, `onscreenWidth`, `onscreenHeight`) from the
//!     parser-scope `(MiSize, MiRow, MiCol, MiRows, MiCols, plane,
//!     subsampling_x, subsampling_y)` tuple. Reuses the §9.3 tables
//!     staged in r148 and the §5.11.46 palette gate constant
//!     [`BLOCK_8X8`] (`3`). Both spec branches handled: the Y branch
//!     returns the §9.3-driven dimensions clipped by `Min(..,
//!     (MiRows - MiRow) * MI_SIZE)` / `Min(.., (MiCols - MiCol) *
//!     MI_SIZE)`, the UV branch applies the `>> subsampling_{x,y}`
//!     shift then the §5.11.49 `<4`-bump (`block_w += 2; onscreen_w
//!     += 2` when post-shift `block_w < 4`, ditto for height).
//!     Walker invariants (`1 <= onscreen_* <= block_*`, `block_* <=
//!     64`) proven over every palette-eligible `(MiSize, sub_x,
//!     sub_y, plane)` combination. Bridges the data-flow gap pinned
//!     by the r147 follow-up test so `read_block` can call
//!     `palette_tokens` once the parser surfaces it. 344 -> 359
//!     tests, zero `#[ignore]`.
//!
//!   * **Round 150.** The §9.3 `Partition_Subsize[ 10 ][ BLOCK_SIZES ]`
//!     lookup (av1-spec p.402-403) plus the §3 enumeration of the
//!     22 named `BLOCK_*` ordinals and the [`BLOCK_INVALID`] (`22`)
//!     sentinel. [`PARTITION_SUBSIZE`] transcribes the table
//!     verbatim — every rectangular `bSize` column carries
//!     `BLOCK_INVALID` per the av1-spec p.401 note "The table will
//!     never get accessed for rectangular block sizes". The typed
//!     accessor [`partition_subsize`] folds the sentinel into
//!     `Option<usize>` (`None` on `BLOCK_INVALID` or out-of-range
//!     indices) so the upcoming §5.11.4 `decode_partition()` body
//!     can read `subSize = Partition_Subsize[ partition ][ bSize ]`
//!     /  `splitSize = Partition_Subsize[ PARTITION_SPLIT ][ bSize ]`
//!     and never hand a sentinel to its recursive children. Tests
//!     cover row-3 (SPLIT) halving identity, row-1 (HORZ) / row-2
//!     (VERT) halving, row-4..7 (`_A` / `_B`) row equality with
//!     rows 1 / 2, row-8..9 (`_4`) quarter-splits, the
//!     `BLOCK_4X4`-only-resolves-for-`PARTITION_NONE` column-0
//!     invariant, exhaustive rectangular-`bSize`-is-invalid
//!     coverage, and the §5.11.4 subdivision-shrinks-area
//!     invariant. The 22 `BLOCK_*` constants and `BLOCK_INVALID`
//!     are also pinned against their av1-spec p.171-172 / p.7
//!     ordinals. 359 -> 375 tests, zero `#[ignore]`.
//!
//!   * **Round 151.** The §5.11.4 `decode_partition( r, c, bSize )`
//!     body (av1-spec p.61-62) — the recursive partition-tree walker
//!     that stitches r137-r145's §9.4 partition-default CDFs,
//!     r145's §8.3.2 [`split_or_horz_cdf`] / [`split_or_vert_cdf`]
//!     binary-CDF derivation, r150's §9.3 [`PARTITION_SUBSIZE`]
//!     table + [`partition_subsize`] accessor, and the §9.3
//!     [`MI_WIDTH_LOG2`] / [`NUM_4X4_BLOCKS_WIDE`] tables into one
//!     [`PartitionWalker`] type. The walker carries the §6.10.4
//!     `MiSizes[r][c]` grid (filled at every leaf via the block's
//!     [`NUM_4X4_BLOCKS_WIDE`] / [`NUM_4X4_BLOCKS_HIGH`] footprint),
//!     and consults it for the §8.3.2 [`partition_ctx`] derivation
//!     `above = AvailU && (Mi_Width_Log2[ MiSizes[r-1][c] ] < bsl)` /
//!     `left  = AvailL && (Mi_Height_Log2[ MiSizes[r][c-1] ] < bsl)`
//!     on every recursive child (av1-spec p.362). The walker emits a
//!     [`Vec<DecodedBlockRecord>`] of `(MiRow, MiCol, MiSize)` leaves
//!     in §5.11.4 syntax order; the actual §5.11.5 `decode_block()`
//!     body remains out of scope and is the next round's target.
//!     All four §5.11.4 edge-of-frame branches are honoured: the
//!     `r >= MiRows || c >= MiCols` early return; the `bSize <
//!     BLOCK_8X8` short-circuit to `PARTITION_NONE` with no symbol
//!     read; the `hasCols`-alone branch that reads `split_or_horz`;
//!     the `hasRows`-alone branch that reads `split_or_vert`; and the
//!     `!hasRows && !hasCols` fall-through to `PARTITION_SPLIT`. All
//!     ten partition arms (`NONE` / `HORZ` / `VERT` / `SPLIT` /
//!     `HORZ_A` / `HORZ_B` / `VERT_A` / `VERT_B` / `HORZ_4` /
//!     `VERT_4`) dispatch the spec's literal `decode_block` /
//!     recursive `decode_partition` calls with the appropriate
//!     `subSize` (the `Partition_Subsize[ partition ][ bSize ]`
//!     lookup) or `splitSize` (the `Partition_Subsize[ PARTITION_SPLIT
//!     ][ bSize ]` lookup). New [`TileGeometry`] type carries the
//!     four §5.11.1 mi-unit tile bounds for the [§5.11.51
//!     `is_inside`](TileGeometry::is_inside) test. The §5.11.4
//!     bottom-right edge clip on the optional `HORZ_4` / `VERT_4`
//!     fourth leaf (`r + quarterBlock4x4 * 3 < MiRows` /
//!     `c + quarterBlock4x4 * 3 < MiCols`) is applied literally.
//!     Out-of-range / out-of-tile inputs surface as
//!     [`Error::PartitionWalkOutOfRange`] (caller-bug only — a
//!     conformant driver never produces one). Tests cover: single-leaf
//!     `PARTITION_NONE` for sub-`BLOCK_8X8` block sizes (no symbol
//!     read); single-leaf and split-emission against the W128
//!     superblock; recursive descent down to `BLOCK_4X4` leaves; all
//!     ten partition arms via deterministic CDF setups; the
//!     edge-of-frame `hasRows`-alone / `hasCols`-alone branches via a
//!     small mi-cropped frame; the §6.10.4 `MiSizes[]` grid-fill
//!     invariant (every cell of the leaf's `bh4 * bw4` footprint
//!     carries `sub_size`); and the §5.11.4 `r >= MiRows ||
//!     c >= MiCols` early-return invariant. The walker leaves the
//!     `decode_block` coefficient / motion-vector / reconstruction
//!     path entirely to the next round.
//!
//!   * **Round 152.** The §5.11.11 `read_skip()` syntax element
//!     (av1-spec p.65) — the per-block `skip` syntax read. Lands as
//!     a new [`PartitionWalker::decode_skip`] method on the r151
//!     walker, plus a `Skips[r][c]` flag grid carried alongside the
//!     existing §6.10.4 `MiSizes[]` grid. Both spec branches are
//!     honoured: when the caller passes `seg_skip_active = true`
//!     (the combined precondition `SegIdPreSkip &&
//!     seg_feature_active( SEG_LVL_SKIP )` from §5.9.14
//!     segmentation state, computed upstream by the frame parser),
//!     the spec short-circuits to `skip = 1` with no symbol read;
//!     otherwise an `S()` symbol is decoded against
//!     `TileSkipCdf[ctx]` where `ctx = AvailU *
//!     Skips[MiRow-1][MiCol] + AvailL * Skips[MiRow][MiCol-1]` per
//!     av1-spec p.378, driven through the walker's `Skips[]` grid
//!     and [`TileGeometry::is_inside`] for `AvailU` / `AvailL`. The
//!     §5.11.5 grid-fill (av1-spec p.65 footer) stamps the decoded
//!     value over the block's `bw4 * bh4` footprint, clipped at the
//!     frame's `MiRows` / `MiCols` extent. New
//!     [`PartitionWalker::skips`] accessor returns a row-major view
//!     of the grid for downstream §5.11.x consumers. Out-of-range
//!     `mi_row` / `mi_col` / `sub_size` surface as
//!     [`Error::PartitionWalkOutOfRange`]. Tests cover: seg
//!     short-circuit no-symbol-read invariant; else-branch S()
//!     against rigged 2-symbol CDFs (forced 0 and 1); footprint
//!     grid-stamp on both branches; ctx-0 selection at the frame
//!     origin (`AvailU = AvailL = false`); ctx-2 selection through
//!     two prior `Skips=1` neighbours; AvailL-false at the
//!     left-tile-column boundary; non-zero tile origin clearing
//!     both `AvailU` / `AvailL`; right-edge `bw4` clip; out-of-range
//!     guard returns; and the initial all-zero `Skips[]` invariant.
//!     394 -> 405 tests, zero `#[ignore]`.
//!
//!   * **Round 154.** The §5.11.10 `read_skip_mode()` syntax element
//!     (av1-spec p.67) — the per-block `skip_mode` syntax read.
//!     Lands as a new [`PartitionWalker::decode_skip_mode`] method
//!     on the r152 walker, plus a `SkipModes[r][c]` flag grid
//!     carried alongside the r152 `Skips[]` and the existing
//!     §6.10.4 `MiSizes[]` grids. The §5.11.10 short-circuit set
//!     (any-true ⇒ `skip_mode = 0`, no symbol read) is honoured:
//!     `seg_feature_active(SEG_LVL_SKIP / REF_FRAME / GLOBALMV)`
//!     collapsed into the caller-provided `seg_skip_mode_off`;
//!     `!skip_mode_present` via the §5.9.21 frame-header scalar;
//!     and `Block_Width[MiSize] < 8 || Block_Height[MiSize] < 8`
//!     derived locally from `sub_size` via the §9.3
//!     [`cdf::block_width`] / [`cdf::block_height`] tables.
//!     Otherwise an `S()` symbol is decoded against
//!     `TileSkipModeCdf[ctx]` with `ctx = AvailU *
//!     SkipModes[MiRow-1][MiCol] + AvailL *
//!     SkipModes[MiRow][MiCol-1]` per av1-spec p.378, routed
//!     through the existing [`skip_mode_ctx`] helper. The §5.11.5
//!     grid-fill stamps the value over the block's `bw4 * bh4`
//!     footprint, clipped at the frame's `MiRows` / `MiCols`
//!     extent. New [`PartitionWalker::skip_modes`] accessor
//!     returns a row-major view. `skip_mode` is the inter-frame
//!     compound-reference shortcut read in §5.11.18
//!     `inter_frame_mode_info` before the rest of the inter
//!     mode decode (intra-only frames never call this). Tests
//!     cover: seg short-circuit; `skip_mode_present` false
//!     short-circuit; both Block_Width-and-Block_Height < 8
//!     short-circuits via BLOCK_4X8 and BLOCK_8X4; else-branch
//!     S() against rigged 2-symbol CDFs (forced 0 and 1);
//!     footprint grid-stamp; ctx-0 selection at the frame
//!     origin; ctx-1 single-neighbour and ctx-2 both-neighbour
//!     selection; non-zero tile origin clearing AvailU / AvailL;
//!     right-edge `bw4` clip; out-of-range guard returns; and
//!     the initial all-zero `SkipModes[]` invariant. 405 -> 417
//!     tests, zero `#[ignore]`.
//!
//!   * **Round 155.** The §5.11.12 `read_delta_qindex()` syntax
//!     element (av1-spec p.67) — the per-superblock quantiser
//!     index delta read. Lands as a new
//!     [`PartitionWalker::decode_delta_qindex`] method on the r154
//!     walker, plus a `CurrentQIndex` scalar tracked across calls
//!     via [`PartitionWalker::current_q_index`] /
//!     [`PartitionWalker::set_current_q_index`]. Honours the
//!     §5.11.12 superblock-skip short-circuit (`MiSize == sbSize
//!     && skip` ⇒ no symbol read; `sbSize` derived from the
//!     §5.5.1 `use_128x128_superblock` flag) and the outer
//!     `ReadDeltas` (§6.10.4) gate. Otherwise an `S()` symbol is
//!     decoded against `TileDeltaQCdf` (a single-row §8.3.2 CDF,
//!     no context index, length `DELTA_Q_SMALL + 2 = 5`); a
//!     decoded value of `DELTA_Q_SMALL = 3` triggers the §5.11.12
//!     escape ladder (`delta_q_rem_bits` `L(3)` + post-increment +
//!     `delta_q_abs_bits` `L(rem_bits + 1)`), reconstructing the
//!     absolute value via `delta_q_abs = delta_q_abs_bits + (1 <<
//!     n) + 1` over the extended range `0..=2 ∪ 3..=511`. Non-zero
//!     `delta_q_abs` reads `delta_q_sign_bit` `L(1)` and applies the
//!     spec's `CurrentQIndex = Clip3(1, 255, CurrentQIndex +
//!     (reducedDeltaQIndex << delta_q_res))` update. New constant
//!     [`cdf::DELTA_Q_SMALL`] (= 3) plus new table
//!     [`cdf::DEFAULT_DELTA_Q_CDF`] transcribed verbatim from §9.4
//!     p.431 (`[28160, 32120, 32677, 32768, 0]`); new field
//!     [`cdf::TileCdfContext::delta_q`] + accessor
//!     [`cdf::TileCdfContext::delta_q_cdf`]. Tests cover: default
//!     CDF literal match; init-from-defaults invariant; sb-skip
//!     short-circuit for both `use_128x128_superblock` values;
//!     ReadDeltas false short-circuit; zero `delta_q_abs` no-update;
//!     literal-positive paths with and without shift; Clip3
//!     lower-bound via hostile seed; Clip3 upper-bound;
//!     DELTA_Q_SMALL escape ladder minimum value; escape ladder
//!     stays in `Clip3(1, 255)` range; cross-call accumulation;
//!     fresh-walker initial `CurrentQIndex = 0`; out-of-range
//!     guard; arithmetic decoder zero-byte sign-bit observation.
//!     417 -> 433 tests, zero `#[ignore]`.
//!
//!   * **Round 156.** The §5.11.13 `read_delta_lf()` syntax element
//!     (av1-spec p.68) — the per-superblock loop-filter delta read,
//!     structurally parallel to §5.11.12 but iterating
//!     `frameLfCount` times over a four-slot `DeltaLF[ i ]`
//!     accumulator with the `delta_lf_multi` flag selecting between
//!     the §8.3.2 single-LF (`TileDeltaLFCdf`) and per-edge multi-LF
//!     (`TileDeltaLFMultiCdf[ i ]`) cdf branches. Lands as a new
//!     [`PartitionWalker::decode_delta_lf`] method plus a
//!     `current_delta_lf: [i32; FRAME_LF_COUNT]` accumulator on the
//!     walker (accessors [`PartitionWalker::current_delta_lf`] and
//!     [`PartitionWalker::reset_current_delta_lf`] for the §5.11.2
//!     tile-entry reset). Honours the §5.11.13 superblock-skip
//!     short-circuit identically to §5.11.12, plus the
//!     `ReadDeltas && delta_lf_present` outer gate (two AND-ed
//!     flags). When the gate passes, `frameLfCount` is derived
//!     locally: `delta_lf_multi == 0` ⇒ 1; `delta_lf_multi == 1 &&
//!     mono_chrome` ⇒ `FRAME_LF_COUNT - 2 = 2`; otherwise
//!     `FRAME_LF_COUNT = 4`. Each iteration reads `delta_lf_abs`
//!     `S()` against the branch-selected CDF, then either the
//!     literal value or the §5.11.13 escape ladder
//!     (`delta_lf_rem_bits` `L(3)` + post-increment +
//!     `delta_lf_abs_bits` `L(rem_bits + 1)` ⇒ `deltaLfAbs =
//!     abs_bits + (1 << n) + 1`), then for non-zero magnitudes
//!     `delta_lf_sign_bit` `L(1)` followed by the
//!     `DeltaLF[ i ] = Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
//!     DeltaLF[ i ] + (reducedDeltaLfLevel << delta_lf_res))`
//!     update. New constants [`cdf::DELTA_LF_SMALL`] (= 3),
//!     [`cdf::FRAME_LF_COUNT`] (= 4), and
//!     [`cdf::MAX_LOOP_FILTER`] (= 63 as `i32` — distinct from the
//!     pre-existing `uncompressed_header_tail::MAX_LOOP_FILTER`
//!     `i16` twin used by §5.9.11). New table
//!     [`cdf::DEFAULT_DELTA_LF_CDF`] transcribed verbatim from §9.4
//!     p.431 (`[28160, 32120, 32677, 32768, 0]`, equal to
//!     `DEFAULT_DELTA_Q_CDF` per the spec listing — preserved as
//!     two independent constants so adaptation drift on one does
//!     not leak through the other). New fields
//!     [`cdf::TileCdfContext::delta_lf`] +
//!     [`cdf::TileCdfContext::delta_lf_multi`] with accessors
//!     [`cdf::TileCdfContext::delta_lf_cdf`] /
//!     [`cdf::TileCdfContext::delta_lf_multi_cdf`]. 17 new cdf-module
//!     tests (433 -> 450): default-CDF literal match (incl.
//!     §9.4 equality with `DEFAULT_DELTA_Q_CDF`); init-from-defaults
//!     invariant for both single-LF and all four multi-LF rows;
//!     sb-skip short-circuit (64x64 and 128x128); `ReadDeltas`
//!     false short-circuit; `delta_lf_present` false short-circuit;
//!     single-LF branch writes only `DeltaLF[0]`; multi-LF colour
//!     branch writes all four slots; multi-LF monochrome branch
//!     writes only the two Y slots; zero-`delta_lf_abs` no-update;
//!     literal-positive with shift; Clip3 upper-bound at
//!     `MAX_LOOP_FILTER = 63`; Clip3 lower-bound via hostile seed;
//!     `DELTA_LF_SMALL` escape ladder minimum value;
//!     cross-call accumulation; fresh-walker initial accumulator
//!     all-zero + `reset_current_delta_lf` round-trip;
//!     out-of-range guard. `decode_av1` / `encode_av1` continue to
//!     return `Error::NotImplemented`.
//!
//!   * **Round 157.** The §5.11.56 `read_cdef()` syntax element
//!     (av1-spec p.104) — the per-leaf CDEF-index read, plus the
//!     §5.11.55 `clear_cdef()` per-superblock sentinel reset. Lands
//!     as new [`PartitionWalker::decode_cdef`] +
//!     [`PartitionWalker::clear_cdef`] methods on the r156 walker,
//!     plus a `cdef_idx: Vec<i8>` row-major grid sized `MiRows ×
//!     MiCols` with [`PartitionWalker::cdef_idx`] read accessor. CDEF
//!     operates on 64×64 anchor cells, so the walker masks the leaf
//!     `(MiRow, MiCol)` to the anchor at `(MiRow & cdefMask4, MiCol &
//!     cdefMask4)` (`cdefMask4 = ~(cdefSize4 - 1)`, `cdefSize4 =
//!     Num_4x4_Blocks_Wide[ BLOCK_64X64 ] = 16`) and uses the
//!     anchor's `-1` sentinel to decide whether the first leaf in
//!     this anchor should read an `L(cdef_bits)` literal (`cdef_bits`
//!     in `0..=3` per §5.9.19 `f(2)`). When the literal is read, the
//!     grid-fill loop stamps the value across the leaf's
//!     `(w4, h4)` footprint at the `cdefSize4 = 16` stride so super-64
//!     blocks (`BLOCK_128X128`) reach all four anchor cells while
//!     sub-64 blocks touch only their containing anchor. Subsequent
//!     leaves in the same 64×64 anchor short-circuit (no read; the
//!     anchor already holds the value). The §5.11.56 short-circuit
//!     set is honoured: `skip || CodedLossless || !enable_cdef ||
//!     allow_intrabc` ⇒ no read, anchor sentinel/value returned
//!     unchanged. `clear_cdef( r, c, use_128x128_superblock )` —
//!     called by the §5.11.2 tile walk before each superblock —
//!     stamps `-1` at the one (64×64 superblocks) or four (128×128
//!     superblocks) anchor cells; out-of-grid anchors are silently
//!     skipped (the bottom/right superblock can straddle the frame
//!     edge). 18 new cdf-module tests (450 → 468): fresh-walker
//!     all-`-1` invariant; `clear_cdef` 64x64 single-anchor stamp;
//!     `clear_cdef` 128x128 four-anchor stamp; `clear_cdef`
//!     out-of-grid skip; each of the four `skip` / `CodedLossless`
//!     / `!enable_cdef` / `allow_intrabc` short-circuit gates;
//!     first-leaf-reads-literal-and-stamps-anchor; second-leaf-in-
//!     anchor-no-read; `cdef_bits == 0` zero-bit stamp; `cdef_bits ==
//!     3` upper-bound; anchor-mask routes (10, 13) ⇒ (0, 0);
//!     BLOCK_128X128 stamps all four anchors; grid-fill clips at
//!     frame edge; short-circuit returns prior stamp; `clear_cdef`
//!     after stamp resets anchor; out-of-range guard (4-way:
//!     `mi_row` / `mi_col` / `sub_size` / `cdef_bits > 3`).
//!     `decode_av1` / `encode_av1` continue to return
//!     `Error::NotImplemented`.
//!
//!   * **Round 158.** The §5.11.20 `read_is_inter()` syntax element
//!     (av1-spec p.71-72) — the per-block intra/inter classifier read
//!     inside §5.11.18 `inter_frame_mode_info` (after
//!     `inter_segment_id` / `read_skip` / `read_cdef` /
//!     `read_delta_qindex` / `read_delta_lf`) that dispatches
//!     between §5.11.22 `intra_block_mode_info` and §5.11.23
//!     `inter_block_mode_info`. Lands as a new
//!     [`PartitionWalker::decode_is_inter`] method on the r157
//!     walker, plus an `IsInters[r][c]` flag grid carried alongside
//!     the r157 `cdef_idx[]`, r154 `SkipModes[]`, r152 `Skips[]`,
//!     and the existing §6.10.4 `MiSizes[]` grids
//!     (`IsInters[ r + y ][ c + x ] = is_inter` per the §5.11.5
//!     footer at av1-spec p.65). All four arms of the §5.11.20
//!     dispatch are honoured in spec order (first match fires, no
//!     read on the short-circuit arms): Arm 1 — `skip_mode == 1`
//!     forces `is_inter = 1`; Arm 2 —
//!     `seg_feature_active(SEG_LVL_REF_FRAME)` routes through the
//!     caller-pre-computed `FeatureData[segment_id][SEG_LVL_REF_FRAME]
//!     != INTRA_FRAME` boolean (the walker stays
//!     segmentation-state-free, identical to r154's
//!     `seg_skip_mode_off` pattern); Arm 3 —
//!     `seg_feature_active(SEG_LVL_GLOBALMV)` forces `is_inter = 1`;
//!     Arm 4 — `S()` symbol read against `TileIsInterCdf[ctx]` with
//!     `ctx` from the existing [`is_inter_ctx`] helper. The §8.3.2
//!     ctx samples neighbour intra-ness from the complement of the
//!     walker's `IsInters[]` grid (`intra = !is_inter`), with an
//!     unavailable neighbour treated as intra per §5.11.18
//!     (`LeftRefFrame[0] = AvailL ? RefFrames[..][0] : INTRA_FRAME`
//!     ⇒ `None` to [`is_inter_ctx`]). The §5.11.5 grid-fill stamps
//!     the value over the block's `bw4 * bh4` footprint, clipped at
//!     the frame's `MiRows` / `MiCols` extent. New
//!     [`PartitionWalker::is_inters`] accessor returns a row-major
//!     view. 15 new cdf-module tests (468 → 483): fresh-walker grid
//!     all-zero; Arm 1 skip_mode short-circuit (position-invariant
//!     on a hostile `0xFF` buffer); Arm 2 routing to intra and to
//!     inter (both position-invariant); Arm 3 globalmv short-circuit;
//!     Arm 1 takes precedence over Arm 2 / Arm 3; Arm 2 takes
//!     precedence over Arm 3; else-branch S() returning symbol 0 / 1
//!     on rigged binary CDF rows (symbol-1 verifies the footprint
//!     grid-stamp); ctx-0 / ctx-1 / ctx-2 / ctx-3 selection through
//!     intra/inter seed combinations and a `mi_col_start = 4` tile
//!     origin; bottom-right edge clip on `BLOCK_16X16 @ (2, 2)` in
//!     a 4×4 frame; three-way out-of-range guard ⇒
//!     `PartitionWalkOutOfRange`. `decode_av1` / `encode_av1`
//!     continue to return `Error::NotImplemented`.
//!
//!   * **Round 159.** The §5.11.9 `read_segment_id()` syntax element
//!     (av1-spec p.66) — the per-block segment-id reader called
//!     inside §5.11.8 `intra_segment_id` and §5.11.19
//!     `inter_segment_id` whenever `segmentation_enabled`. Lands as
//!     a new [`PartitionWalker::decode_segment_id`] method on the
//!     r158 walker, plus a `SegmentIds[r][c]` grid carried alongside
//!     the r158 `IsInters[]`, r156 `cdef_idx[]`, r154 `SkipModes[]`,
//!     r152 `Skips[]`, and the existing §6.10.4 `MiSizes[]` grids
//!     (`SegmentIds[ r + y ][ c + x ] = segment_id` per the §5.11.5
//!     grid-fill footer). The grid is pre-filled with the §5.11.9
//!     `-1` sentinel; cells inside a decoded block's `bh4 * bw4`
//!     footprint then carry `segment_id` in `0..MAX_SEGMENTS = 0..8`.
//!     The §5.11.9 neighbour cascade is honoured exactly as
//!     spelled out: `prevUL` requires both `AvailU` AND `AvailL`;
//!     `prevU` and `prevL` each gate on their own edge; out-of-grid
//!     neighbours fall through to `-1`. The four-arm `pred`
//!     derivation (`prevU == -1` ⇒ `prevL / 0`; `prevL == -1` ⇒
//!     `prevU`; `prevUL == prevU` ⇒ `prevU`; else `prevL`) is
//!     preserved verbatim. The §5.11.9 dispatch distinguishes two
//!     paths: `skip != 0` ⇒ `segment_id = pred` (zero bits read);
//!     else `diff S()` against `TileSegmentIdCdf[ctx]` (ctx from
//!     the existing [`segment_id_ctx`] helper) then
//!     `segment_id = neg_deinterleave(diff, pred, last_active_seg_id + 1)`.
//!     The walker stays segmentation-state-free: the caller passes
//!     `last_active_seg_id` (the §5.9.14 trailing derivation) and
//!     the `skip` value the §5.11.11 `decode_skip` just returned.
//!     New public module-level [`neg_deinterleave`] helper
//!     transcribes the §5.11.9 bijection. New
//!     [`PartitionWalker::segment_ids`] accessor returns a row-major
//!     view. 11 new cdf-module tests (483 → 494): fresh-walker grid
//!     all `-1`; skip short-circuit at frame origin writes
//!     `segment_id = pred = 0` (no `S()` bit consumed on a hostile
//!     `0xFF` byte buffer); skip inherits `prev_u` when `prev_l` is
//!     unavailable; non-skip path with `pred = 0` returns `diff`
//!     unchanged; direct `neg_deinterleave` table exercises for both
//!     `2 * ref < max` and `2 * ref >= max` branches plus edge cases
//!     (`ref == 0` identity, `ref == max - 1` inverted, smallest
//!     non-trivial alphabet `max = 2`); ctx-0 origin selection;
//!     ctx-2 all-neighbours-match selection through three
//!     walker-stamped seeds; bottom-right edge clip on
//!     `BLOCK_16X16 @ (2, 2)` in a 4×4 frame; four-way out-of-range
//!     guard. `decode_av1` / `encode_av1` continue to return
//!     `Error::NotImplemented`.
//!
//!   * **Round 160.** The §5.11.8 `intra_segment_id()` syntax
//!     element (av1-spec p.66) — the intra-frame variant of the
//!     per-block segment-id read, called from §5.11.7
//!     `intra_frame_mode_info` on both the `SegIdPreSkip` and
//!     `!SegIdPreSkip` arms. Lands as a new
//!     [`PartitionWalker::decode_intra_segment_id`] method that
//!     dispatches on the caller-passed `segmentation_enabled` bit:
//!     when set, descends into r159
//!     [`PartitionWalker::decode_segment_id`] (which does the
//!     §5.11.9 neighbour cascade + skip / non-skip dispatch + grid
//!     fill); when clear, stamps `segment_id = 0` over the leaf's
//!     `bh4 * bw4` footprint without reading any bits. Both arms
//!     then resolve the §5.11.8 `Lossless = LosslessArray[
//!     segment_id ]` lookup via a caller-supplied
//!     `lossless_array: &[bool; MAX_SEGMENTS]` (the §6.8.2
//!     per-segment table the frame-header walk computes from
//!     `qindex` + `DeltaQ?Dc` + `DeltaQ?Ac`). Returns
//!     `(segment_id, lossless)`. The walker stays
//!     segmentation-state-free: callers pass `segmentation_enabled`,
//!     `last_active_seg_id`, and `lossless_array` per-call (matching
//!     the r159 pattern). Range guards (out-of-range `sub_size`,
//!     `mi_row` / `mi_col` past the frame's mi extent,
//!     `last_active_seg_id >= MAX_SEGMENTS`) fire on both arms so
//!     the no-symbol path is total over the same input space as the
//!     bitstream-reading path. 7 new cdf-module tests (494 → 501):
//!     `segmentation_enabled = false` no-read on a hostile `0xFF`
//!     buffer with `lossless_array[0] = true` and `= false` arms;
//!     `segmentation_enabled = true, skip = 1` at frame origin
//!     (pred = 0, no `S()` consumed); `segmentation_enabled = true,
//!     skip = 0` reading `diff = 3` on a rigged CDF and looking up
//!     `lossless_array[3]`; per-segment Lossless indexing (rig
//!     `diff = 5`, set `lossless_array[5] = false` while every
//!     other slot is `true`, expect `Lossless = false`);
//!     bottom-right edge-clip on `BLOCK_16X16 @ (2, 2)` in a 4×4
//!     frame on the `!segmentation_enabled` arm; five-way
//!     out-of-range guard covering both arms. `decode_av1` /
//!     `encode_av1` continue to return `Error::NotImplemented`.
//!
//!   * **Round 161.** The §5.11.7 `intra_frame_mode_info()`
//!     **prefix dispatcher** (av1-spec p.64) — the per-block top-level
//!     entry-point that composes the §5.11.7 first 11 lines:
//!     `skip = 0`; conditional pre-skip `intra_segment_id()`;
//!     `skip_mode = 0`; `read_skip()`; conditional post-skip
//!     `intra_segment_id()`; `read_cdef()`; `read_delta_qindex()`;
//!     `read_delta_lf()`; the fixed `ReadDeltas = 0` / `RefFrame[0]
//!     = INTRA_FRAME` / `RefFrame[1] = NONE` assignments. Lands as
//!     a new [`PartitionWalker::decode_intra_frame_mode_info_prefix`]
//!     method on the r160 walker, returning the new public
//!     [`cdf::IntraFrameModeInfoPrefix`] struct carrying every
//!     post-call observable (`skip`, `skip_mode`, `segment_id`,
//!     `lossless`, `cdef_idx`, `current_q_index`,
//!     `current_delta_lf`, `ref_frame`). The §5.11.7 `SegIdPreSkip`
//!     conditional routes the §5.11.8 call before or after the
//!     §5.11.11 `read_skip()` per the caller-passed
//!     `seg_id_pre_skip` boolean (the §5.9.14 trailing
//!     derivation). `skip_mode` is fixed at `0` because the
//!     intra-frame walk never calls `decode_skip_mode` (§5.11.10
//!     short-circuits on `!skip_mode_present` for intra-only frames
//!     per §5.9.21). The §6.10.4 `ReadDeltas = 0` assignment is
//!     left to the caller (the walker remains stateless about
//!     per-superblock first-block detection), matching the
//!     §6.10.4 pattern existing
//!     [`PartitionWalker::decode_delta_qindex`] /
//!     [`PartitionWalker::decode_delta_lf`] call sites already
//!     use. Range guards (out-of-range `sub_size`, `mi_row` /
//!     `mi_col`, `last_active_seg_id >= MAX_SEGMENTS`, `cdef_bits >
//!     3`) fire on the dispatcher level before any inner read so a
//!     caller bug never produces a partial-read. 8 new cdf-module
//!     tests (501 → 509): minimum-bit path (one `S()` consumed for
//!     `read_skip`); `SegIdPreSkip = true` arm reading
//!     `segment_id` first then `skip`; `SegIdPreSkip = false` arm
//!     post-skip with `skip = 1` triggering the §5.11.9
//!     short-circuit; seg-skip-active forcing `skip = 1` with zero
//!     bits consumed on a hostile `0xFF` buffer; fixed
//!     `ref_frame = [INTRA_FRAME, NONE]`; `read_deltas` true
//!     wiring through to both `delta_q` and `delta_lf` reads;
//!     five-way out-of-range guard; `skip_mode` always `0` on both
//!     pre-skip arms (verifies `SkipModes[]` grid untouched). The
//!     §5.11.7 follow-on body (`use_intrabc` arm + `intra_block_mode_info`
//!     composite) and the §5.11.18 `inter_frame_mode_info` /
//!     §5.11.19 `inter_segment_id` two-call protocol remain the
//!     next round's targets. `decode_av1` / `encode_av1` continue
//!     to return `Error::NotImplemented`.
//!
//!   * **Round 162.** The §5.11.19 `inter_segment_id( preSkip )`
//!     syntax element (av1-spec p.71) — the inter-frame variant of
//!     the per-block segment-id read, called twice per block from
//!     §5.11.18 (once before §5.11.11 `read_skip()` with
//!     `preSkip = 1` and once after with `preSkip = 0`). Lands as
//!     a new [`PartitionWalker::decode_inter_segment_id`] method.
//!     New `SEGMENT_ID_PREDICTED_CONTEXTS = 3` (§9.3) constant and
//!     [`cdf::DEFAULT_SEGMENT_ID_PREDICTED_CDF`] table verbatim from
//!     §9.4 (av1-spec p.442 — three uniform `[128 * 128, 32768, 0]`
//!     rows for the binary `seg_id_predicted` symbol). New
//!     [`cdf::TileCdfContext::segment_id_predicted`] field
//!     initialised from `DEFAULT_SEGMENT_ID_PREDICTED_CDF` plus the
//!     [`cdf::TileCdfContext::segment_id_predicted_cdf`] selector
//!     implementing the §8.3.2 `TileSegmentIdPredictedCdf[ ctx ]`
//!     index. New persistent `above_seg_pred_context` (length
//!     `mi_cols`) and `left_seg_pred_context` (length `mi_rows`)
//!     buffers on `PartitionWalker` per the §8.3.1 tile-entry
//!     initialisation; the §8.3.2 ctx walk reads `LeftSegPredContext[
//!     MiRow ] + AboveSegPredContext[ MiCol ]` (each `0..=1`; sum
//!     `0..=2 < SEGMENT_ID_PREDICTED_CONTEXTS`).
//!     [`PartitionWalker::above_seg_pred_context`] /
//!     [`PartitionWalker::left_seg_pred_context`] accessors surface
//!     the arrays. The dispatcher routes the full §5.11.19 cascade
//!     exactly: outer `!segmentation_enabled` collapses to
//!     `segment_id = 0`; inner `!segmentation_update_map` adopts
//!     `predictedSegmentId`; `pre_skip && !SegIdPreSkip` early-exit
//!     returns `segment_id = 0`; the post-skip `skip != 0` arm
//!     zeroes the context arrays then descends into `decode_segment_id`
//!     (the §5.11.9 skip short-circuit fires inside); the
//!     `segmentation_temporal_update == 1` arm reads
//!     `seg_id_predicted`, branches to `predictedSegmentId` adopt
//!     or `read_segment_id()`, and stamps the context arrays with
//!     the just-read flag; the `temporal_update == 0` fall-through
//!     reads `read_segment_id()` without touching the context arrays
//!     (per spec). `predicted_segment_id` (§5.11.21 `get_segment_id()`)
//!     is caller-supplied from the §6.10 reference-frame walk so
//!     the walker stays inter-frame-state-free. Range guards
//!     (out-of-range `sub_size`, `mi_row`/`mi_col` past extent,
//!     `last_active_seg_id >= MAX_SEGMENTS`, and the new
//!     `predicted_segment_id > last_active_seg_id` invariant) fire
//!     up-front on every arm. 11 new cdf-module tests (509 → 520):
//!     fresh-walker context-array zeroing; `!segmentation_enabled`
//!     no-read on both pre/post-skip arms; `!segmentation_update_map`
//!     predicted-id adoption; `pre_skip && !SegIdPreSkip` early-exit;
//!     post-skip + `skip != 0` zeroing context arrays then descending
//!     into the §5.11.9 short-circuit (poisoned context arrays prove
//!     the spec write fires); `temporal_update == 1` + rigged
//!     `seg_id_predicted = 1` adopting predicted id and stamping
//!     context to `1`; `temporal_update == 1` + rigged
//!     `seg_id_predicted = 0` descending into `decode_segment_id`
//!     and stamping context to `0`; `temporal_update == 0`
//!     fall-through leaving context untouched; five-way out-of-range
//!     guard; `Default_Segment_Id_Predicted_Cdf` layout; the
//!     `segment_id_predicted_cdf` accessor round-trip. The §5.11.18
//!     `inter_frame_mode_info()` top-level dispatcher remains the
//!     next round's architectural payoff. `decode_av1` /
//!     `encode_av1` continue to return `Error::NotImplemented`.
//!
//!   * **Round 163.** The §5.11.21 `get_segment_id()` pure-helper
//!     function (av1-spec p.72) — the inter-frame per-block segment-id
//!     **prediction** lookup that scans the previous frame's
//!     `PrevSegmentIds[][]` over the on-screen window covered by the
//!     current block and returns the smallest id found. Lands as a
//!     new free function [`cdf::get_segment_id`] (re-exported at the
//!     crate root) that takes the previous-frame segmentation surface
//!     as a row-major `&[i32]` slice plus its (`prev_mi_rows`,
//!     `prev_mi_cols`) extent, the current frame's mi-extent
//!     (`mi_rows`, `mi_cols`) and anchor (`mi_row`, `mi_col`), and the
//!     block's `sub_size` (§3 `BLOCK_*` ordinal). Returns
//!     `Some(seg)` with the §5.11.21 `Min` reduction result over
//!     `xMis = Min(MiCols - MiCol, bw4)` × `yMis = Min(MiRows - MiRow,
//!     bh4)` cells; `seg` is in `-1..=7` (the `-1` sentinel surfaces
//!     when an unwritten previous-frame cell falls inside the window,
//!     letting callers detect a malformed reference surface via the
//!     existing §5.11.19 `predicted_segment_id > last_active_seg_id`
//!     range guard). Returns `None` for caller-bug arguments
//!     (out-of-range `sub_size`, anchor outside the current frame,
//!     previous-frame extent smaller than the current frame's, or
//!     `prev_segment_ids` length not matching `prev_mi_rows *
//!     prev_mi_cols`). The function is **pure** (no walker state, no
//!     bitreader, no CDF) and complements the r162
//!     [`cdf::PartitionWalker::decode_inter_segment_id`] caller —
//!     which takes a pre-computed `predicted_segment_id: u8` so the
//!     walker can stay inter-frame-state-free — by giving the §6.10
//!     reference-frame walk a verbatim spec-shaped routine to compute
//!     that argument from `PrevSegmentIds[]`. 12 new cdf-module
//!     tests (520 → 532): uniform-0 / uniform-7 reductions over
//!     several block sizes; explicit Min-over-2x2-window cells with
//!     out-of-window decoy values that must not contribute; frame-edge
//!     `xMis`/`yMis` clipping on a `BLOCK_16X16` anchored at the
//!     bottom-right of a 4x4 mi-grid; `-1` sentinel round-trip; a
//!     wider-than-current previous frame exercising the
//!     `prev_mi_cols` row stride; single-cell `BLOCK_4X4` covering
//!     exactly `prev[MiRow][MiCol]` with a neighbour-cell decoy;
//!     out-of-range guards for invalid `sub_size`, anchor past frame
//!     extent, previous-frame extent smaller than current frame's,
//!     and length / shape mismatch; an end-to-end composition test
//!     feeding `get_segment_id`'s result into
//!     `decode_inter_segment_id`'s no-read
//!     `!segmentation_update_map` arm and verifying the predicted id
//!     is adopted with zero bit reads on a hostile `0xFF` buffer.
//!     The §5.11.18 `inter_frame_mode_info()` top-level dispatcher,
//!     §5.11.7 `use_intrabc` arm, and §5.11.22 `intra_block_mode_info`
//!     composite remain the next round's targets. `decode_av1` /
//!     `encode_av1` continue to return `Error::NotImplemented`.
//!
//!   * **Round 168.** §5.11.17 `read_var_tx_size()` (av1-spec p.70) +
//!     §5.11.18 `inter_frame_mode_info()` (av1-spec p.71) — the two
//!     missing inter-arm composites that bound the §5.11.5 walker's
//!     inter side. The §5.11.17 reader is exposed as
//!     [`cdf::PartitionWalker::read_var_tx_size`] (callable
//!     standalone) and wired into the §5.11.16 inter-arm so
//!     [`cdf::PartitionWalker::read_block_tx_size`] no longer
//!     surfaces [`Error::ReadVarTxSizeUnsupported`] on the
//!     `TX_MODE_SELECT && is_inter && !skip && !Lossless` arm — the
//!     recursion stamps `InterTxSizes[]` per terminal-else leaf and
//!     the outer §5.11.5 footer stamps `TxSizes[]` over the full
//!     block footprint. The §8.3.2 `txfm_split` ctx selector inlines
//!     the spec's `get_above_tx_width` / `get_left_tx_height`
//!     helpers as private [`cdf::PartitionWalker`] methods against
//!     the walker's `Skips[]` / `IsInters[]` / `MiSizes[]` /
//!     `InterTxSizes[]` grids, and routes the
//!     `(above, left, txSzSqrUp, maxTxSz)` derivation through the
//!     existing [`cdf::txfm_split_ctx`] helper. New free function
//!     [`cdf::find_tx_size`] (§5.11.36 helper, re-exported) supports
//!     the `maxTxSz = find_tx_size(size, size)` step of the ctx
//!     formula.
//!
//!     The §5.11.18 reader is exposed as
//!     [`cdf::PartitionWalker::decode_inter_frame_mode_info`]
//!     composing every leaf already wired through the walker:
//!     [`cdf::PartitionWalker::decode_inter_segment_id`] (pre- and
//!     post-skip arms), [`cdf::PartitionWalker::decode_skip_mode`],
//!     [`cdf::PartitionWalker::decode_skip`] (gated by
//!     `skip_mode == 0`), [`cdf::PartitionWalker::decode_cdef`],
//!     [`cdf::PartitionWalker::decode_delta_qindex`],
//!     [`cdf::PartitionWalker::decode_delta_lf`], and
//!     [`cdf::PartitionWalker::decode_is_inter`]. The terminal
//!     `if (is_inter)` dispatch short-circuits at two new `Error`
//!     variants: [`Error::InterBlockModeInfoUnsupported`] (§5.11.23
//!     `inter_block_mode_info()` — the next-round target for the
//!     MV stack / ref-frame readers) and
//!     [`Error::IntraBlockModeInfoUnsupported`] (§5.11.22
//!     `intra_block_mode_info()` — the next-round target for the
//!     per-block intra angle / UV mode readers). New
//!     [`cdf::DecodedInterFrameModeInfo`] aggregate carries every
//!     §5.11.18 derived value.
//!
//!     The §5.11.5 [`cdf::PartitionWalker::decode_block_syntax`]
//!     walker is unchanged on the `frame_is_intra = false` arm — it
//!     still surfaces [`Error::DecodeBlockInterFrameUnsupported`]
//!     (the umbrella stub) because the §5.11.18 reader needs
//!     additional caller state (`segmentation_update_map` /
//!     `segmentation_temporal_update` / `predicted_segment_id` /
//!     `seg_*_active` overrides / `skip_mode_present`) the §5.11.5
//!     driver doesn't yet thread through. Direct callers of
//!     `decode_inter_frame_mode_info` get the full pre-dispatch walk
//!     and the §5.11.22 / §5.11.23 distinction.
//!
//!     11 new integration tests cover the §5.11.17 base case + depth
//!     cap + split path + frame-edge clip + caller-bug guards + the
//!     wired §5.11.16 inter-arm, plus the §5.11.18 intra / inter
//!     stub dispatch + skip-mode-forces-skip + segment-globalmv
//!     arm + caller-bug guards + the `DecodedInterFrameModeInfo`
//!     public-API smoke. Plus 1 new unit test for `find_tx_size`.
//!     `decode_av1` / `encode_av1` continue to return
//!     [`Error::NotImplemented`].
//!
//!   * **Round 167.** The §5.11.16 `read_block_tx_size()` reader
//!     (av1-spec p.70) — the per-block transform-size syntax-tree
//!     read that r166's §5.11.5 walker hit as the first stub on the
//!     intra arm. Lands as a new
//!     [`cdf::PartitionWalker::read_block_tx_size`] method
//!     (callable standalone) and is wired into the §5.11.5 walker
//!     so the walker now reaches
//!     [`Error::DecodeBlockComputePredictionUnsupported`] (§5.11.30)
//!     instead.
//!
//!     The reader transcribes the spec body one-to-one. The outer
//!     `TX_MODE_SELECT && MiSize > BLOCK_4X4 && is_inter && !skip &&
//!     !Lossless` arm routes to §5.11.17 `read_var_tx_size` (the
//!     next-arc target — surfaces a new
//!     [`Error::ReadVarTxSizeUnsupported`]). The `else` arm inlines
//!     §5.11.15 `read_tx_size(!skip || !is_inter)`: a `Lossless`
//!     short-circuit forces `TxSize = TX_4X4`; otherwise `TxSize`
//!     starts at `Max_Tx_Size_Rect[ MiSize ]` and is further split
//!     via `Split_Tx_Size[]` for `tx_depth` iterations when the
//!     §5.11.15 `MiSize > BLOCK_4X4 && allowSelect && TxMode ==
//!     TX_MODE_SELECT` gate fires. The §8.3.2 `tx_depth` ctx walks
//!     the neighbour `IsInters[]` / `MiSizes[]` / `InterTxSizes[]`
//!     ladder per `get_above_tx_width` / `get_left_tx_height` and
//!     selects the CDF row via `Max_Tx_Depth[ MiSize ]`. The
//!     `else`-arm grid-fill stamps both
//!     [`cdf::PartitionWalker::inter_tx_sizes`] and the §5.11.5
//!     footer's [`cdf::PartitionWalker::tx_sizes`] across the block
//!     footprint.
//!
//!     New spec tables ([`cdf::MAX_TX_SIZE_RECT`] /
//!     [`cdf::MAX_TX_DEPTH_TABLE`] / [`cdf::SPLIT_TX_SIZE`] /
//!     [`cdf::MAX_VARTX_DEPTH`]) and 14 new rectangular `TX_*`
//!     ordinals ([`cdf::TX_4X8`] / ... / [`cdf::TX_64X16`])
//!     transcribe av1-spec p.402, p.69, p.404, §3 p.7, and §6.10.16
//!     respectively. New [`cdf::PartitionWalker`] grids
//!     `tx_sizes` / `inter_tx_sizes` cover §5.11.5 / §5.11.16
//!     write semantics. [`cdf::DecodedBlock`] gains a `tx_size`
//!     field; [`cdf::PartitionWalker::decode_block_syntax`] gains a
//!     `tx_mode_select: bool` parameter (threaded from the
//!     §5.9.21 / §6.8.21 frame-header `TxMode` derivation).
//!
//!     10 new integration tests cover the new reader's spec paths;
//!     5 new unit tests verify the spec tables.
//!     `decode_av1` / `encode_av1` continue to return
//!     [`Error::NotImplemented`].
//!
//!   * **Round 166.** The §5.11.5 `decode_block()` syntax-walker
//!     skeleton (av1-spec p.63-64) — the missing dispatcher that
//!     the §5.11.4 partition walker recurses into at every leaf.
//!     Lands as a new [`cdf::PartitionWalker::decode_block_syntax`]
//!     method that performs the §5.11.5 prologue (`MiRow`, `MiCol`,
//!     `MiSize`, `bw4`, `bh4`; the `HasChroma` three-arm dispatch
//!     on `bh4 == 1 && subsampling_y && (MiRow & 1) == 0` and
//!     `bw4 == 1 && subsampling_x && (MiCol & 1) == 0`; `AvailU`,
//!     `AvailL`, `AvailUChroma`, `AvailLChroma` derivation with the
//!     spec's chroma fix-up arm) and the §5.11.6 `mode_info()`
//!     intra arm composed from r161's
//!     [`cdf::PartitionWalker::decode_intra_frame_mode_info_prefix`],
//!     r164's [`cdf::PartitionWalker::decode_use_intrabc`], and
//!     r165's [`cdf::PartitionWalker::decode_intra_frame_y_mode`].
//!     The §5.11.49 `palette_tokens()` call is a no-op on the
//!     no-palette path. Short-circuits at the §5.11.16
//!     `read_block_tx_size()` call with a new
//!     [`Error::DecodeBlockReadBlockTxSizeUnsupported`] — the
//!     immediate next-round target.
//!
//!     Four new [`Error`] variants surface the §5.11.5 next-round
//!     boundaries one-to-one:
//!     [`Error::DecodeBlockInterFrameUnsupported`] (§5.11.18
//!     `inter_frame_mode_info()`, fires on `frame_is_intra = false`
//!     with zero bits consumed);
//!     [`Error::DecodeBlockReadBlockTxSizeUnsupported`] (§5.11.16,
//!     the stub the walker actually reaches);
//!     [`Error::DecodeBlockComputePredictionUnsupported`] (§5.11.30,
//!     reserved for the round that lands §5.11.16); and
//!     [`Error::DecodeBlockResidualUnsupported`] (§5.11.34,
//!     reserved for the round that lands §5.11.30).
//!
//!     New [`cdf::DecodedBlock`] per-block aggregate — publicly
//!     constructible — carries the §5.11.5 prologue + §5.11.7
//!     mode-info derivations on the no-stub path. Reachable once
//!     §5.11.16 lands.
//!
//!     §5.11.4 partition driver mirror:
//!     [`cdf::PartitionWalker::decode_partition_syntax`] walks the
//!     same recursion shape as
//!     [`cdf::PartitionWalker::decode_partition`] but routes every
//!     leaf through `decode_block_syntax` instead of the leaf-only
//!     emitter. Stub propagates from the first leaf that fires it.
//!
//!     10 new integration tests
//!     (`tests/decode_block_syntax_walker.rs`) cover the §5.11.5
//!     prologue's three-arm `HasChroma` dispatch on `BLOCK_4X4`,
//!     the §5.11.6 inter-arm stub, the §5.11.7 `SegIdPreSkip`
//!     pre-skip arm reaching the §5.11.16 stub after the composed
//!     reads fire in spec order, out-of-range guards, the §5.11.4
//!     partition driver's `BLOCK_4X4` short-circuit arm propagating
//!     the stub, the §5.11.4 `r >= MiRows` line-1 early return,
//!     `DecodedBlock` public-API constructibility, `BLOCK_8X16`
//!     grid-fill footprint (bw4 = 2, bh4 = 4), and `BLOCK_16X16`
//!     with `cdef_bits = 2` exercising the §5.11.56 literal-bits
//!     read.
//!
//!     The §5.11.7 follow-on `else`-arm elements
//!     (`intra_angle_info_y`, `uv_mode`, `read_cfl_alphas`,
//!     `intra_angle_info_uv`, `palette_mode_info`,
//!     `filter_intra_mode_info`) and the `use_intrabc == 1`
//!     MV-stack / assign-mv body remain bounded leaf targets that
//!     can be slotted into a future round before or alongside
//!     §5.11.16. `decode_av1` / `encode_av1` continue to return
//!     [`Error::NotImplemented`].
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
    block_height, block_width, cfl_alpha_u_ctx, cfl_alpha_v_ctx, coeff_cdf_q_ctx,
    compound_mode_ctx, compute_tx_type, find_tx_size, get_br_ctx, get_coeff_base_ctx,
    get_coeff_base_eob_ctx, get_palette_color_context, get_segment_id, get_tx_class,
    inter_tx_type_set, interintra_ctx, interp_filter_ctx, intra_dir, intra_mode_ctx,
    intra_tx_type_set, is_inter_ctx, is_tx_type_in_set, mi_height_log2, mi_width_log2, mv_ctx,
    neg_deinterleave, num_4x4_blocks_high, num_4x4_blocks_wide,
    palette_color_context_from_neighbors, palette_color_ctx, palette_tokens_args,
    palette_tokens_plane, palette_uv_mode_ctx, palette_y_mode_ctx, partition_ctx,
    partition_subsize, ref_count_ctx, segment_id_ctx, size_group, skip_ctx, skip_mode_ctx,
    split_or_horz_cdf, split_or_vert_cdf, tx_depth_ctx, txfm_split_ctx, DecodedBlock,
    DecodedBlockRecord, DecodedInterFrameModeInfo, IntraFrameModeInfoPrefix, PaletteColorContext,
    PalettePlane, PaletteTokensArgs, PartitionWalker, TileCdfContext, TileGeometry,
    ADJUSTED_TX_SIZE, ADST_ADST, ADST_DCT, ADST_FLIPADST, BLOCK_128X128, BLOCK_128X64, BLOCK_16X16,
    BLOCK_16X32, BLOCK_16X4, BLOCK_16X64, BLOCK_16X8, BLOCK_32X16, BLOCK_32X32, BLOCK_32X64,
    BLOCK_32X8, BLOCK_4X16, BLOCK_4X4, BLOCK_4X8, BLOCK_64X128, BLOCK_64X16, BLOCK_64X32,
    BLOCK_64X64, BLOCK_8X16, BLOCK_8X32, BLOCK_8X4, BLOCK_8X8, BLOCK_INVALID, BLOCK_SIZES,
    BLOCK_SIZE_GROUPS, BR_CDF_SIZE, BWD_REFS, CFL_ALPHABET_SIZE, CFL_ALPHA_CONTEXTS,
    CFL_JOINT_SIGNS, CLASS0_SIZE, COEFF_BASE_CTX_OFFSET, COEFF_BASE_POS_CTX_OFFSET,
    COEFF_BASE_RANGE, COEFF_CDF_Q_CTXS, COMPOUND_IDX_CONTEXTS, COMPOUND_MODES,
    COMPOUND_MODE_CONTEXTS, COMPOUND_MODE_CTX_MAP, COMPOUND_TYPES, COMP_GROUP_IDX_CONTEXTS,
    COMP_INTER_CONTEXTS, COMP_NEWMV_CTXS, COMP_REF_TYPE_CONTEXTS, DCT_ADST, DCT_DCT, DCT_FLIPADST,
    DC_SIGN_CONTEXTS, DEFAULT_ANGLE_DELTA_CDF, DEFAULT_CFL_ALPHA_CDF, DEFAULT_CFL_SIGN_CDF,
    DEFAULT_COEFF_BASE_CDF, DEFAULT_COEFF_BASE_EOB_CDF, DEFAULT_COEFF_BR_CDF,
    DEFAULT_COMPOUND_IDX_CDF, DEFAULT_COMPOUND_MODE_CDF, DEFAULT_COMPOUND_TYPE_CDF,
    DEFAULT_COMP_BWD_REF_CDF, DEFAULT_COMP_GROUP_IDX_CDF, DEFAULT_COMP_MODE_CDF,
    DEFAULT_COMP_REF_CDF, DEFAULT_COMP_REF_TYPE_CDF, DEFAULT_DC_SIGN_CDF, DEFAULT_DELTA_LF_CDF,
    DEFAULT_DELTA_Q_CDF, DEFAULT_DRL_MODE_CDF, DEFAULT_EOB_EXTRA_CDF, DEFAULT_EOB_PT_1024_CDF,
    DEFAULT_EOB_PT_128_CDF, DEFAULT_EOB_PT_16_CDF, DEFAULT_EOB_PT_256_CDF, DEFAULT_EOB_PT_32_CDF,
    DEFAULT_EOB_PT_512_CDF, DEFAULT_EOB_PT_64_CDF, DEFAULT_FILTER_INTRA_CDF,
    DEFAULT_FILTER_INTRA_MODE_CDF, DEFAULT_INTERP_FILTER_CDF, DEFAULT_INTER_INTRA_CDF,
    DEFAULT_INTER_INTRA_MODE_CDF, DEFAULT_INTER_TX_TYPE_SET1_CDF, DEFAULT_INTER_TX_TYPE_SET2_CDF,
    DEFAULT_INTER_TX_TYPE_SET3_CDF, DEFAULT_INTRABC_CDF, DEFAULT_INTRA_FRAME_Y_MODE_CDF,
    DEFAULT_INTRA_TX_TYPE_SET1_CDF, DEFAULT_INTRA_TX_TYPE_SET2_CDF, DEFAULT_IS_INTER_CDF,
    DEFAULT_MOTION_MODE_CDF, DEFAULT_MV_BIT_CDF, DEFAULT_MV_CLASS0_BIT_CDF,
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
    DEFAULT_REF_MV_CDF, DEFAULT_SEGMENT_ID_CDF, DEFAULT_SEGMENT_ID_PREDICTED_CDF,
    DEFAULT_SINGLE_REF_CDF, DEFAULT_SKIP_CDF, DEFAULT_SKIP_MODE_CDF, DEFAULT_TXB_SKIP_CDF,
    DEFAULT_TXFM_SPLIT_CDF, DEFAULT_TX_16X16_CDF, DEFAULT_TX_32X32_CDF, DEFAULT_TX_64X64_CDF,
    DEFAULT_TX_8X8_CDF, DEFAULT_UNI_COMP_REF_CDF, DEFAULT_UV_MODE_CFL_ALLOWED_CDF,
    DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF, DEFAULT_WEDGE_INDEX_CDF, DEFAULT_WEDGE_INTER_INTRA_CDF,
    DEFAULT_Y_MODE_CDF, DEFAULT_ZERO_MV_CDF, DELTA_LF_SMALL, DELTA_Q_SMALL, DIRECTIONAL_MODES,
    DRL_MODE_CONTEXTS, EOB_COEF_CONTEXTS, EXT_PARTITION_TYPES, FILTER_INTRA_MODE_TO_INTRA_DIR,
    FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, FRAME_LF_COUNT, FWD_REFS, H_ADST, H_DCT,
    H_FLIPADST, IDTX, INTERINTRA_MODES, INTERP_FILTERS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_NONE,
    INTER_TX_TYPE_SET1_SIZES, INTER_TX_TYPE_SET3_SIZES, INTRA_FILTER_MODES, INTRA_MODES,
    INTRA_MODE_CONTEXT, INTRA_MODE_CONTEXTS, INTRA_TX_TYPE_SET1_SIZES, INTRA_TX_TYPE_SET2_SIZES,
    IS_INTER_CONTEXTS, LEVEL_CONTEXTS, MAG_REF_OFFSET_WITH_TX_CLASS, MAX_ANGLE_DELTA, MAX_TX_DEPTH,
    MAX_TX_DEPTH_TABLE, MAX_TX_SIZE_RECT, MAX_VARTX_DEPTH, MI_HEIGHT_LOG2, MI_SIZE, MI_SIZE_LOG2,
    MI_WIDTH_LOG2, MODE_TO_TXFM, MOTION_MODES, MV_CLASSES, MV_COMPS, MV_CONTEXTS,
    MV_INTRABC_CONTEXT, MV_JOINTS, MV_OFFSET_BITS, NEW_MV_CONTEXTS, NUM_4X4_BLOCKS_HIGH,
    NUM_4X4_BLOCKS_WIDE, NUM_BASE_LEVELS, PALETTE_BLOCK_SIZE_CONTEXTS, PALETTE_COLORS,
    PALETTE_COLOR_CONTEXT, PALETTE_COLOR_CONTEXTS, PALETTE_COLOR_HASH_MULTIPLIERS,
    PALETTE_MAX_COLOR_CONTEXT_HASH, PALETTE_NUM_NEIGHBORS, PALETTE_SIZES, PALETTE_UV_MODE_CONTEXTS,
    PALETTE_Y_MODE_CONTEXTS, PARTITION_CONTEXTS, PARTITION_HORZ, PARTITION_HORZ_4,
    PARTITION_HORZ_A, PARTITION_HORZ_B, PARTITION_NONE, PARTITION_SPLIT, PARTITION_SUBSIZE,
    PARTITION_TYPES_TOTAL, PARTITION_VERT, PARTITION_VERT_4, PARTITION_VERT_A, PARTITION_VERT_B,
    PLANE_TYPES, REF_CONTEXTS, REF_MV_CONTEXTS, SEGMENT_ID_CONTEXTS, SEGMENT_ID_PREDICTED_CONTEXTS,
    SIG_COEF_CONTEXTS, SIG_COEF_CONTEXTS_2D, SIG_COEF_CONTEXTS_EOB, SIG_REF_DIFF_OFFSET,
    SIG_REF_DIFF_OFFSET_NUM, SINGLE_REFS, SIZE_GROUP, SKIP_CONTEXTS, SKIP_MODE_CONTEXTS,
    SPLIT_TX_SIZE, TXB_SKIP_CONTEXTS, TXFM_PARTITION_CONTEXTS, TX_16X16, TX_16X32, TX_16X4,
    TX_16X64, TX_16X8, TX_32X16, TX_32X32, TX_32X64, TX_32X8, TX_4X16, TX_4X4, TX_4X8, TX_64X16,
    TX_64X32, TX_64X64, TX_8X16, TX_8X32, TX_8X4, TX_8X8, TX_CLASS_2D, TX_CLASS_HORIZ,
    TX_CLASS_VERT, TX_HEIGHT, TX_SET_DCTONLY, TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3,
    TX_SET_INTRA_1, TX_SET_INTRA_2, TX_SET_TYPES_INTER, TX_SET_TYPES_INTRA, TX_SIZES, TX_SIZES_ALL,
    TX_SIZE_CONTEXTS, TX_SIZE_SQR_UP, TX_TYPES, TX_TYPES_INTRA_SET1, TX_TYPES_INTRA_SET2,
    TX_TYPES_SET2, TX_TYPES_SET3, TX_TYPE_IN_SET_INTER, TX_TYPE_IN_SET_INTRA, TX_WIDTH,
    TX_WIDTH_LOG2, UNIDIR_COMP_REFS, UV_INTRA_MODES_CFL_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED,
    V_ADST, V_DCT, V_FLIPADST, V_PRED, WEDGE_TYPES, ZERO_MV_CONTEXTS,
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
    /// The §5.11.49 `palette_tokens` walker
    /// ([`crate::palette_tokens_plane`]) was called with caller-supplied
    /// dimensions / palette size / buffer that violate the spec's
    /// implicit preconditions (palette size outside `2..=PALETTE_COLORS`,
    /// `onscreen_{w,h}` zero or greater than `block_{w,h}`,
    /// `color_index_map >= palette_size`, stride below `block_width`,
    /// or output buffer smaller than `block_height * stride`). The
    /// spec gates the walker behind §5.11.49's `if (PaletteSize{Y,UV})`
    /// guard plus §5.11.x block-size discipline, so a conformant call
    /// site never produces this.
    InvalidPaletteWalkArgs,
    /// During the §5.11.49 palette diagonal walk, the §5.11.50
    /// `ColorContextHash` landed on an unmapped entry of
    /// `Palette_Color_Context[]` (the slots that hold `-1`: hashes
    /// `0`, `1`, `3`, `4`). The spec sweep over every realisable
    /// neighbour combination at every palette size shows these hashes
    /// are unreachable from a conformant decode state, so producing
    /// one is a bitstream-conformance violation (or a buffer-aliasing
    /// bug in the caller's `color_map`).
    PaletteColorContextUnmapped,
    /// The §5.11.4 [`crate::PartitionWalker::decode_partition`] walker
    /// hit an out-of-range `bSize` (>= [`crate::BLOCK_SIZES`]), an
    /// out-of-range `partition`, a `bsl` that did not select a
    /// partition CDF row (a `bsl == 0` or `bsl > 5`), or a
    /// [`crate::partition_subsize`] lookup that returned
    /// [`crate::BLOCK_INVALID`]. Each of these is a caller bug — the
    /// public §5.11.4 entry is gated on a valid `bSize` (the
    /// superblock size, in `BLOCK_8X8..=BLOCK_128X128`) and the
    /// recursive children are constructed from in-range
    /// [`crate::PARTITION_SUBSIZE`] table values, so a conformant
    /// driver never produces one.
    PartitionWalkOutOfRange,
    /// The §5.11.5 [`crate::PartitionWalker::decode_block_syntax`]
    /// walker reached the §5.11.6 `mode_info()` inter-frame arm
    /// (`FrameIsIntra == 0`) — §5.11.18 `inter_frame_mode_info()` is
    /// the next round's target and currently STUBBED. Triggered by
    /// calling the syntax walker with `frame_is_intra = false`. The
    /// keyframe / intra-only path (`frame_is_intra = true`) routes
    /// through the implemented intra arm and never produces this.
    DecodeBlockInterFrameUnsupported,
    /// The §5.11.5 [`crate::PartitionWalker::decode_block_syntax`]
    /// walker reached the §5.11.16 `read_block_tx_size()` call — the
    /// per-block transform-size syntax-tree read. The §5.11.15 /
    /// §5.11.16 / §5.11.17 transform-tree readers are the next
    /// round's target and currently STUBBED. The walker has completed
    /// the §5.11.5 prologue + §5.11.6 `mode_info()` (intra arm) +
    /// §5.11.49 `palette_tokens()` (no-op on the no-palette path) up
    /// to but not including this call.
    ///
    /// As of r167 the variant is retained for API stability but is no
    /// longer produced by `decode_block_syntax` (the §5.11.16 reader
    /// landed). It remains reachable via direct calls into the walker
    /// from out-of-tree consumers that pre-date r167.
    DecodeBlockReadBlockTxSizeUnsupported,
    /// The §5.11.16 [`crate::PartitionWalker::read_block_tx_size`]
    /// reader reached the `TX_MODE_SELECT && MiSize > BLOCK_4X4 &&
    /// is_inter && !skip && !Lossless` arm that loops over
    /// `read_var_tx_size( row, col, maxTxSz, 0 )` per
    /// `(txH4, txW4)` sub-rectangle. §5.11.17 `read_var_tx_size` is
    /// the next round's target and currently STUBBED. Unreachable
    /// from [`crate::PartitionWalker::decode_block_syntax`] because
    /// that walker short-circuits the inter arm at
    /// [`Self::DecodeBlockInterFrameUnsupported`] upstream; surfaces
    /// only when callers drive [`crate::PartitionWalker::read_block_tx_size`]
    /// directly with `is_inter = 1`, `skip = 0`, `lossless = false`,
    /// `tx_mode_select = true`.
    ReadVarTxSizeUnsupported,
    /// The §5.11.5 [`crate::PartitionWalker::decode_block_syntax`]
    /// walker reached the §5.11.30 `compute_prediction()` call — the
    /// per-block intra / inter prediction sample-generation pass.
    /// `compute_prediction()` and its §7.11 sub-routines are the next
    /// round's target and currently STUBBED. Currently unreachable
    /// from `decode_block_syntax` because the walker short-circuits
    /// at [`Error::DecodeBlockReadBlockTxSizeUnsupported`] first; reserved
    /// for the round that lands `read_block_tx_size()` and exposes
    /// this stub.
    DecodeBlockComputePredictionUnsupported,
    /// The §5.11.5 [`crate::PartitionWalker::decode_block_syntax`]
    /// walker reached the §5.11.34 `residual()` call — the per-block
    /// transform-coefficient read + inverse-transform + reconstruction
    /// pass. `residual()` and its §7.12 sub-routines are the next
    /// round's target and currently STUBBED. Currently unreachable
    /// from `decode_block_syntax` because the walker short-circuits
    /// at [`Error::DecodeBlockReadBlockTxSizeUnsupported`] first; reserved
    /// for the round that lands `compute_prediction()` and exposes
    /// this stub.
    DecodeBlockResidualUnsupported,
    /// The §5.11.18 [`crate::PartitionWalker::decode_inter_frame_mode_info`]
    /// walker reached the §5.11.22 `intra_block_mode_info()` call — the
    /// per-block intra-mode-info composite (`y_mode` / `intra_angle_info_y`
    /// / `uv_mode` / `read_cfl_alphas` / `intra_angle_info_uv` /
    /// `palette_mode_info` / `filter_intra_mode_info`). The composite is
    /// reachable from the §5.11.18 `else` arm of `if ( is_inter )`. The
    /// remaining sub-elements (per-block intra angle-delta and UV-mode
    /// readers) are next-round targets; this stub fires after the
    /// §5.11.18 prologue + `read_is_inter` settle on `is_inter == 0`.
    IntraBlockModeInfoUnsupported,
    /// The §5.11.18 [`crate::PartitionWalker::decode_inter_frame_mode_info`]
    /// walker reached the §5.11.23 `inter_block_mode_info()` call — the
    /// per-block inter-mode-info composite (`read_ref_frames` /
    /// `find_mv_stack` / compound-mode / `assign_mv` /
    /// `read_interintra_mode` / `read_motion_mode` / `read_compound_type`
    /// / interpolation-filter reads). The composite is reachable from
    /// the §5.11.18 `if ( is_inter )` arm. The MV-stack / reference-frame
    /// readers are next-round targets; this stub fires after the
    /// §5.11.18 prologue + `read_is_inter` settle on `is_inter == 1`.
    InterBlockModeInfoUnsupported,
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
            Self::InvalidPaletteWalkArgs => write!(
                f,
                "oxideav-av1: §5.11.49 palette_tokens_plane caller-side preconditions violated"
            ),
            Self::PaletteColorContextUnmapped => write!(
                f,
                "oxideav-av1: §5.11.49 palette ColorContextHash maps to an unreachable -1 slot of Palette_Color_Context[]"
            ),
            Self::PartitionWalkOutOfRange => write!(
                f,
                "oxideav-av1: §5.11.4 decode_partition walker hit an out-of-range bSize / partition / bsl / Partition_Subsize lookup"
            ),
            Self::DecodeBlockInterFrameUnsupported => write!(
                f,
                "oxideav-av1: §5.11.5 decode_block reached §5.11.6 mode_info inter-frame arm — §5.11.18 inter_frame_mode_info() pending next round"
            ),
            Self::DecodeBlockReadBlockTxSizeUnsupported => write!(
                f,
                "oxideav-av1: §5.11.5 decode_block reached §5.11.16 read_block_tx_size() — retained for API stability post-r167"
            ),
            Self::ReadVarTxSizeUnsupported => write!(
                f,
                "oxideav-av1: §5.11.16 read_block_tx_size reached §5.11.17 read_var_tx_size() — variable transform-tree recursion pending next round"
            ),
            Self::DecodeBlockComputePredictionUnsupported => write!(
                f,
                "oxideav-av1: §5.11.5 decode_block reached §5.11.30 compute_prediction() — intra / inter prediction sample generation pending next round"
            ),
            Self::DecodeBlockResidualUnsupported => write!(
                f,
                "oxideav-av1: §5.11.5 decode_block reached §5.11.34 residual() — transform-coefficient read + reconstruction pending next round"
            ),
            Self::IntraBlockModeInfoUnsupported => write!(
                f,
                "oxideav-av1: §5.11.18 inter_frame_mode_info reached §5.11.22 intra_block_mode_info() — per-block intra angle / UV mode reads pending next round"
            ),
            Self::InterBlockModeInfoUnsupported => write!(
                f,
                "oxideav-av1: §5.11.18 inter_frame_mode_info reached §5.11.23 inter_block_mode_info() — MV stack / ref-frame readers pending next round"
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
