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
    cfl_alpha_u_ctx, cfl_alpha_v_ctx, coeff_cdf_q_ctx, compound_mode_ctx, compute_tx_type,
    get_br_ctx, get_coeff_base_ctx, get_coeff_base_eob_ctx, get_palette_color_context,
    get_tx_class, inter_tx_type_set, interintra_ctx, interp_filter_ctx, intra_dir, intra_mode_ctx,
    intra_tx_type_set, is_inter_ctx, is_tx_type_in_set, mv_ctx,
    palette_color_context_from_neighbors, palette_color_ctx, palette_tokens_plane,
    palette_uv_mode_ctx, palette_y_mode_ctx, partition_ctx, ref_count_ctx, segment_id_ctx,
    size_group, skip_ctx, skip_mode_ctx, split_or_horz_cdf, split_or_vert_cdf, tx_depth_ctx,
    txfm_split_ctx, PaletteColorContext, PalettePlane, TileCdfContext, ADJUSTED_TX_SIZE, ADST_ADST,
    ADST_DCT, ADST_FLIPADST, BLOCK_128X128, BLOCK_SIZES, BLOCK_SIZE_GROUPS, BR_CDF_SIZE, BWD_REFS,
    CFL_ALPHABET_SIZE, CFL_ALPHA_CONTEXTS, CFL_JOINT_SIGNS, CLASS0_SIZE, COEFF_BASE_CTX_OFFSET,
    COEFF_BASE_POS_CTX_OFFSET, COEFF_BASE_RANGE, COEFF_CDF_Q_CTXS, COMPOUND_IDX_CONTEXTS,
    COMPOUND_MODES, COMPOUND_MODE_CONTEXTS, COMPOUND_MODE_CTX_MAP, COMPOUND_TYPES,
    COMP_GROUP_IDX_CONTEXTS, COMP_INTER_CONTEXTS, COMP_NEWMV_CTXS, COMP_REF_TYPE_CONTEXTS,
    DCT_ADST, DCT_DCT, DCT_FLIPADST, DC_SIGN_CONTEXTS, DEFAULT_ANGLE_DELTA_CDF,
    DEFAULT_CFL_ALPHA_CDF, DEFAULT_CFL_SIGN_CDF, DEFAULT_COEFF_BASE_CDF,
    DEFAULT_COEFF_BASE_EOB_CDF, DEFAULT_COEFF_BR_CDF, DEFAULT_COMPOUND_IDX_CDF,
    DEFAULT_COMPOUND_MODE_CDF, DEFAULT_COMPOUND_TYPE_CDF, DEFAULT_COMP_BWD_REF_CDF,
    DEFAULT_COMP_GROUP_IDX_CDF, DEFAULT_COMP_MODE_CDF, DEFAULT_COMP_REF_CDF,
    DEFAULT_COMP_REF_TYPE_CDF, DEFAULT_DC_SIGN_CDF, DEFAULT_DRL_MODE_CDF, DEFAULT_EOB_EXTRA_CDF,
    DEFAULT_EOB_PT_1024_CDF, DEFAULT_EOB_PT_128_CDF, DEFAULT_EOB_PT_16_CDF, DEFAULT_EOB_PT_256_CDF,
    DEFAULT_EOB_PT_32_CDF, DEFAULT_EOB_PT_512_CDF, DEFAULT_EOB_PT_64_CDF, DEFAULT_FILTER_INTRA_CDF,
    DEFAULT_FILTER_INTRA_MODE_CDF, DEFAULT_INTERP_FILTER_CDF, DEFAULT_INTER_INTRA_CDF,
    DEFAULT_INTER_INTRA_MODE_CDF, DEFAULT_INTER_TX_TYPE_SET1_CDF, DEFAULT_INTER_TX_TYPE_SET2_CDF,
    DEFAULT_INTER_TX_TYPE_SET3_CDF, DEFAULT_INTRA_FRAME_Y_MODE_CDF, DEFAULT_INTRA_TX_TYPE_SET1_CDF,
    DEFAULT_INTRA_TX_TYPE_SET2_CDF, DEFAULT_IS_INTER_CDF, DEFAULT_MOTION_MODE_CDF,
    DEFAULT_MV_BIT_CDF, DEFAULT_MV_CLASS0_BIT_CDF, DEFAULT_MV_CLASS0_FR_CDF,
    DEFAULT_MV_CLASS0_HP_CDF, DEFAULT_MV_CLASS_CDF, DEFAULT_MV_FR_CDF, DEFAULT_MV_HP_CDF,
    DEFAULT_MV_JOINT_CDF, DEFAULT_MV_SIGN_CDF, DEFAULT_NEW_MV_CDF,
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
    DEFAULT_SKIP_MODE_CDF, DEFAULT_TXB_SKIP_CDF, DEFAULT_TXFM_SPLIT_CDF, DEFAULT_TX_16X16_CDF,
    DEFAULT_TX_32X32_CDF, DEFAULT_TX_64X64_CDF, DEFAULT_TX_8X8_CDF, DEFAULT_UNI_COMP_REF_CDF,
    DEFAULT_UV_MODE_CFL_ALLOWED_CDF, DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF, DEFAULT_WEDGE_INDEX_CDF,
    DEFAULT_WEDGE_INTER_INTRA_CDF, DEFAULT_Y_MODE_CDF, DEFAULT_ZERO_MV_CDF, DIRECTIONAL_MODES,
    DRL_MODE_CONTEXTS, EOB_COEF_CONTEXTS, EXT_PARTITION_TYPES, FILTER_INTRA_MODE_TO_INTRA_DIR,
    FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, FWD_REFS, H_ADST, H_DCT, H_FLIPADST, IDTX,
    INTERINTRA_MODES, INTERP_FILTERS, INTERP_FILTER_CONTEXTS, INTERP_FILTER_NONE,
    INTER_TX_TYPE_SET1_SIZES, INTER_TX_TYPE_SET3_SIZES, INTRA_FILTER_MODES, INTRA_MODES,
    INTRA_MODE_CONTEXT, INTRA_MODE_CONTEXTS, INTRA_TX_TYPE_SET1_SIZES, INTRA_TX_TYPE_SET2_SIZES,
    IS_INTER_CONTEXTS, LEVEL_CONTEXTS, MAG_REF_OFFSET_WITH_TX_CLASS, MAX_ANGLE_DELTA, MAX_TX_DEPTH,
    MODE_TO_TXFM, MOTION_MODES, MV_CLASSES, MV_COMPS, MV_CONTEXTS, MV_INTRABC_CONTEXT, MV_JOINTS,
    MV_OFFSET_BITS, NEW_MV_CONTEXTS, NUM_BASE_LEVELS, PALETTE_BLOCK_SIZE_CONTEXTS, PALETTE_COLORS,
    PALETTE_COLOR_CONTEXT, PALETTE_COLOR_CONTEXTS, PALETTE_COLOR_HASH_MULTIPLIERS,
    PALETTE_MAX_COLOR_CONTEXT_HASH, PALETTE_NUM_NEIGHBORS, PALETTE_SIZES, PALETTE_UV_MODE_CONTEXTS,
    PALETTE_Y_MODE_CONTEXTS, PARTITION_CONTEXTS, PARTITION_HORZ, PARTITION_HORZ_4,
    PARTITION_HORZ_A, PARTITION_HORZ_B, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT,
    PARTITION_VERT_4, PARTITION_VERT_A, PARTITION_VERT_B, PLANE_TYPES, REF_CONTEXTS,
    REF_MV_CONTEXTS, SEGMENT_ID_CONTEXTS, SIG_COEF_CONTEXTS, SIG_COEF_CONTEXTS_2D,
    SIG_COEF_CONTEXTS_EOB, SIG_REF_DIFF_OFFSET, SIG_REF_DIFF_OFFSET_NUM, SINGLE_REFS, SIZE_GROUP,
    SKIP_CONTEXTS, SKIP_MODE_CONTEXTS, TXB_SKIP_CONTEXTS, TXFM_PARTITION_CONTEXTS, TX_16X16,
    TX_32X32, TX_4X4, TX_64X64, TX_8X8, TX_CLASS_2D, TX_CLASS_HORIZ, TX_CLASS_VERT, TX_HEIGHT,
    TX_SET_DCTONLY, TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3, TX_SET_INTRA_1, TX_SET_INTRA_2,
    TX_SET_TYPES_INTER, TX_SET_TYPES_INTRA, TX_SIZES, TX_SIZES_ALL, TX_SIZE_CONTEXTS,
    TX_SIZE_SQR_UP, TX_TYPES, TX_TYPES_INTRA_SET1, TX_TYPES_INTRA_SET2, TX_TYPES_SET2,
    TX_TYPES_SET3, TX_TYPE_IN_SET_INTER, TX_TYPE_IN_SET_INTRA, TX_WIDTH, TX_WIDTH_LOG2,
    UNIDIR_COMP_REFS, UV_INTRA_MODES_CFL_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED, V_ADST, V_DCT,
    V_FLIPADST, V_PRED, WEDGE_TYPES, ZERO_MV_CONTEXTS,
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
