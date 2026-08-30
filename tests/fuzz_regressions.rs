//! Regression corpus for inputs the scheduled `Fuzz` workflow found.
//!
//! Each case is a libFuzzer-minimized adversarial input that once
//! panicked the decoder. The contract (same as the `decode` fuzz
//! target): any byte shape must produce `Ok(..)` or a typed
//! [`oxideav_av1::Error`] — never a panic, overflow, or hang.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Byte-counting allocator so the memory-amplification regressions
/// below can assert on the TOTAL bytes a decode requests (peak live
/// size alone hides churn: the r452 finding freed every copy it made,
/// yet the churn is what blew the fuzz runner's RSS ceiling).
struct CountingAlloc;

static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TOTAL_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f` and return the total bytes allocated while it ran (the
/// harness may run other tests in parallel and debug builds allocate
/// more generously than release, so the count is an upper bound on
/// this decode's own requests — the asserted ceilings sit orders of
/// magnitude under the pre-fix figures; the measured decodes are
/// serialized against each other).
fn allocated_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let before = TOTAL_ALLOCATED.load(Ordering::Relaxed);
    let out = f();
    let after = TOTAL_ALLOCATED.load(Ordering::Relaxed);
    (out, after - before)
}

/// Decode raw hex into bytes (test-local helper; fixtures embed hex so
/// the suite runs in per-crate CI without a fixture checkout).
fn hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "hex literal must have even length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// 2026-07-03 scheduled-Fuzz crash `8f12c488…`: a truncated IVF frame
/// whose coefficient payload ends inside the §5.11.39 golomb chain.
/// The §8.2.2 arithmetic decoder then pads with zero bits forever, so
/// the uncapped `do { length++ } while ( !golomb_length_bit )` loop
/// of §5.11.39 spun ~2^32 iterations and overflowed `length`
/// (`attempt to add with overflow` at the `length += 1`). Fixed by
/// the 30-bit robustness cap surfacing
/// [`oxideav_av1::Error::GolombLengthOverflow`].
#[test]
fn golomb_length_chain_is_bounded_on_truncated_coefficients() {
    let bytes = hex(
        "444b494600002000000020443f4946cccc57cc7acccccccccccccc4b491c0000\
         2000000028cccccccc9e55af0e46095f4cf7ffd1ff46f6001f00097affff0000\
         000000c52206fff7ff286a00",
    );
    // Success and every typed error are both acceptable; only a panic
    // or a hang (caught by the test harness / CI timeout) is a
    // finding. This input specifically must fail fast rather than
    // walk a multi-billion-iteration zero-bit tail.
    let start = std::time::Instant::now();
    let _ = oxideav_av1::decode_av1(&bytes);
    assert!(
        start.elapsed() < std::time::Duration::from_secs(30),
        "decode of a 76-byte adversarial input must not take tens of seconds"
    );
}

/// 2026-07-03 dispatched-Fuzz crash `c055549e…` (surfaced once the
/// golomb cap above unblocked deeper coverage): a §5.9.30
/// `film_grain_params()` whose chroma point-count `f(4)` literal codes
/// a value past the `MAX_NUM_CHROMA_POINTS = 10` conformance bound,
/// indexing past the fixed-size point arrays (`index out of bounds:
/// the len is 10 but the index is 10`). Fixed by the §5.9.30
/// conformance checks surfacing
/// [`oxideav_av1::Error::FilmGrainPointCountOverflow`].
#[test]
fn film_grain_point_counts_are_bounded() {
    let bytes = hex(
        "444b494600002000002900000000000000fff8cccc33cc23cccc0800000000002c\
         000000cc28cccc4a61a04b0e4d095d1e00ff01823b00051f001c0a00003100000e\
         2000f7ffa2a9b61f4400fffff8cce001000204ffff00",
    );
    // Success and every typed error are both acceptable; only a panic
    // is a finding.
    let _ = oxideav_av1::decode_av1(&bytes);
}

/// 2026-07-11 scheduled-Fuzz crash `c25ecb93…`: a multi-tile frame
/// header combined with a §5.11.1 tile-group prologue whose
/// `tile_start_and_end_present_flag == 1` carries `tg_start > tg_end`.
/// The §5.11.1 walk derived the tile span as `tg_end - tg_start` and
/// panicked with `attempt to subtract with overflow`. Fixed by the
/// §6.10.1 conformance reject (`tg_start <= tg_end < NumTiles`) in
/// `parse_tile_group_obu_body`.
#[test]
fn tile_group_start_beyond_end_is_rejected_not_panicking() {
    let bytes = hex(
        "444b494600002000000020444b4946cc8ccccccccccccccccccccc4b49460000\
         20000000282000025031364b0e460900000100c52a3c00613600870000000000\
         0000d6364c00000000006a00",
    );
    let _ = oxideav_av1::decode_av1(&bytes);
}

/// 2026-07-11 scheduled-Fuzz crash `99736d56…` (red 2026-07-11 through
/// 2026-07-16): a frame header whose §5.9.14 `segmentation_params()`
/// derives `LastActiveSegId < 7`, combined with a §5.11.9
/// `read_segment_id()` `S()` payload coding a symbol past that bound.
/// The `S()` read runs against the full 8-symbol
/// `Default_Segment_Id_Cdf` regardless of `LastActiveSegId`, so the
/// attacker-chosen arithmetic payload forced `diff >
/// last_active_seg_id` and tripped the `§5.11.9 diff is in
/// 0..=last_active_seg_id` debug assertion (and, past it, would have
/// driven `neg_deinterleave` outside its `diff < max` domain). Fixed
/// by the §6.10.8 conformance reject surfacing
/// [`oxideav_av1::Error::SegmentIdOutOfRange`].
#[test]
fn segment_id_symbol_past_last_active_seg_id_is_rejected() {
    let bytes = hex(
        "444b494600002000000020444b49464c8ccccccccccccccccccccc4b49420000\
         20000000284c00cc4a9f554b0e46090000010046003c666137008f0000002800\
         200000000000000000020000",
    );
    // Success and every typed error are both acceptable; only a panic
    // is a finding.
    let _ = oxideav_av1::decode_av1(&bytes);
}

/// 2026-08-29 scheduled-Fuzz `oom-578e9910…` (`libFuzzer:
/// out-of-memory (used: 2058Mb; limit: 2048Mb)`): a 76-byte
/// still-picture whose KEY frame is 2572 × 281. The frame decodes
/// fine — the finding was memory AMPLIFICATION, not a hostile
/// reservation: the §5.11.40 `compute_tx_type()` neighbour lookup
/// took a full COPY of the `MiRows * MiCols` `TxTypes[]` grid per
/// transform block, so a frame of N 4×4 blocks requested N grid
/// copies (46 KiB × 44 573 blocks ≈ 2.07 GB of churn off 76 input
/// bytes; the 1024 × 8065 sibling below churned 267 GB). Under the
/// sanitizer's quarantine that churn is the runner's RSS. Fixed by
/// reading the grid through a shared borrow.
#[test]
fn per_transform_block_tx_type_lookup_does_not_copy_the_grid() {
    let bytes = hex(
        "444b4946000020000000000000000000004ccccccccccccccccccc4b49460000\
         2000000028cccccccc2f554b0e46095ff64a0b46004900d03700114c00000000\
         6f000001370000000027ffcc",
    );
    let (result, allocated) = allocated_during(|| oxideav_av1::decode_av1(&bytes));
    let frames = result.expect("the 2572x281 still picture decodes");
    assert_eq!(frames.len(), 1);
    // Pre-fix: 2 066 760 864 bytes in ~46 KiB grid copies alone. The
    // legitimate reservation (planes + mi grids for 722 892 luma
    // samples) measures ~38 MB in release, ~190 MB in a debug build.
    assert!(
        allocated < 512 << 20,
        "76-byte input requested {allocated} bytes — per-block grid copies are back"
    );
}

/// 2026-08-30 local `slow-unit-80700107…` reproduced from the same
/// run's corpus: a 92-byte still picture declaring 1024 × 8065
/// (516 608 4×4 blocks) that hit the same per-block `TxTypes[]` copy —
/// 267 GB requested in a 3.6 s decode (28 s under the sanitizer).
/// The payload ends with a size-field-less OBU so the decode itself
/// returns [`oxideav_av1::Error::MissingSizeField`] after the frame.
/// The second half of the test is the r452 Annex A picture-size
/// gate: at a 2^21-sample ceiling (the fuzz harness setting) the
/// header is rejected before any frame-sized buffer is reserved.
#[test]
fn tall_still_picture_decodes_without_grid_churn_and_respects_the_cap() {
    let bytes = hex(
        "444b494600002000008200004e41000000000000000000ffffffff5c0008004b\
         3000000010000e0e0e0e0e0e0e0e0e5cff00ffff00f70000000038373737d808\
         0800810000000000000201000000000000000000f700444b4900004e",
    );
    let start = std::time::Instant::now();
    let (result, allocated) = allocated_during(|| oxideav_av1::decode_av1(&bytes));
    assert!(matches!(result, Err(oxideav_av1::Error::MissingSizeField)));
    assert!(
        allocated < 4 << 30,
        "92-byte input requested {allocated} bytes — per-block grid copies are back"
    );
    assert!(
        start.elapsed() < std::time::Duration::from_secs(30),
        "decode of a 92-byte adversarial input must not take tens of seconds"
    );

    // 1024 * 8065 = 8 258 560 luma samples > 2^21: rejected up front,
    // and the reject costs no frame-sized reservation (the IVF walk
    // plus header parse stay under a megabyte).
    let mut session = oxideav_av1::decoder::SpecDecodeSession::new();
    session.set_max_picture_size(1 << 21);
    let (capped, capped_alloc) = allocated_during(|| session.decode_ivf(&bytes));
    assert!(matches!(
        capped,
        Err(oxideav_av1::Error::PictureSizeExceedsLimit)
    ));
    assert!(
        capped_alloc < 1 << 20,
        "a capped reject must not reserve frame storage (requested {capped_alloc} bytes)"
    );

    // The library default is Annex A's largest defined MaxPicSize
    // (levels 6.0–6.3) and admits this 8.26 M-sample frame.
    assert_eq!(oxideav_av1::decoder::MAX_PICTURE_SIZE, 35_651_584);
    assert_eq!(
        oxideav_av1::decoder::SpecDecodeSession::new().max_picture_size(),
        oxideav_av1::decoder::MAX_PICTURE_SIZE
    );
}
