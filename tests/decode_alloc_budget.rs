//! r475 — still-image decode budgets: the allocation budget. The
//! serial decode's live-allocation high-water mark and allocation
//! count stay within a per-sample budget (a counting global
//! allocator — this binary holds exactly one test, so nothing else
//! allocates during the measurement): the pre-r475 driver held three
//! frame-sized `i32` copies plus a second copy of the output.

use oxideav_av1::encoder::still::{StillOptions, encode_still_yuv};
use oxideav_av1::encoder::yuv_frame::{ChromaFormat, YuvFrame};
use oxideav_av1::registry::make_decoder;
use oxideav_core::{CodecParameters, Frame, Packet, TimeBase};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            System.alloc(layout)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            System.dealloc(ptr, layout)
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            let old = layout.size();
            if new_size > old {
                let live = LIVE.fetch_add(new_size - old, Ordering::Relaxed) + (new_size - old);
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(old - new_size, Ordering::Relaxed);
            }
            System.realloc(ptr, layout, new_size)
        }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// A textured picture: slow gradients, a diagonal ridge pattern and
/// a pseudo-random grain, so the encoder keeps real coefficients and
/// elects the in-loop filters.
fn textured(width: u32, height: u32, format: ChromaFormat) -> YuvFrame {
    let mut f = YuvFrame::filled(width, height, 8, format, 0);
    let mut seed = 0x1234_5678u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as i32 & 15
    };
    for y in 0..height as usize {
        for x in 0..width as usize {
            let g = (x * 3 / 7 + y * 2 / 5) % 256;
            let ridge = if ((x + y) / 13) % 3 == 0 { 40 } else { 0 };
            let v = (g as i32 + ridge + noise() - 8).clamp(0, 255);
            f.y[y * width as usize + x] = v as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for y in 0..ch {
        for x in 0..cw {
            f.u[y * cw + x] = ((x * 5 / 3 + y) % 200 + 28) as u16;
            f.v[y * cw + x] = ((x + y * 7 / 4) % 180 + 40) as u16;
        }
    }
    f
}

fn encode_tiled(frame: &YuvFrame) -> Vec<u8> {
    let opts = StillOptions {
        tile_cols_log2: 1,
        tile_rows_log2: 1,
        ..StillOptions::new(100)
    };
    encode_still_yuv(frame, &opts)
        .expect("still encode")
        .temporal_unit_bytes
}

fn decode_registry(tu: &[u8]) -> Vec<Vec<u8>> {
    let params = CodecParameters::video("av1".into());
    let mut dec = make_decoder(&params).expect("make_decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), tu.to_vec()))
        .expect("send_packet");
    dec.flush().expect("flush");
    let mut planes = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => planes.extend(v.planes.into_iter().map(|p| p.data)),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore) => continue,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    }
    assert!(!planes.is_empty(), "no frame decoded");
    planes
}

#[test]
fn serial_decode_stays_within_the_allocation_budget() {
    let frame = textured(1024, 768, ChromaFormat::Yuv420);
    let tu = encode_tiled(&frame);
    let samples = 1024usize * 768;
    ALLOCS.store(0, Ordering::Relaxed);
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    let base = LIVE.load(Ordering::Relaxed);
    let out = decode_registry(&tu);
    let peak = PEAK.load(Ordering::Relaxed) - base;
    let allocs = ALLOCS.load(Ordering::Relaxed);
    drop(out);
    // r475 measured 13.4 bytes / luma sample at 12 MP (the `i32`
    // `CurrFrame` is 6 of them, the per-mi grids ~3.5, the output
    // bytes 1.5) and 0.06 allocations / sample; the budget leaves
    // headroom for the fixed costs of a 0.8 MP picture.
    let peak_budget = 24 * samples + (8 << 20);
    let alloc_budget = samples / 4 + 50_000;
    assert!(
        peak <= peak_budget,
        "peak live {peak} bytes exceeds the budget {peak_budget}"
    );
    assert!(
        allocs <= alloc_budget,
        "{allocs} allocations exceed the budget {alloc_budget}"
    );
}
