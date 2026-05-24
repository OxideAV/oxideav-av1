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
//! The remaining ~90 `Default_*_Cdf` arrays of §9.4 (the y_mode,
//! uv_mode, angle-delta, tx-size, coefficient, palette, … groups),
//! the `init_coeff_cdfs` coefficient tables, and the §8.3.2
//! `split_or_horz` / `split_or_vert` / `tx_depth` / `txfm_split` / …
//! selections are a clear followup: each is a mechanical transcription
//! of one §9.4 table plus its §8.3.2 paragraph, slotted into the same
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
}
