//! Progressive §7.20 reference planes (r481).
//!
//! Under a worker budget a frame's samples become final band by band
//! while its walk and in-loop filters are still running, and a later
//! frame's motion compensation only reads the rows its blocks' motion
//! vectors (plus the §7.11.3.4 filter taps) reach. [`RefPlanes`] holds
//! the stored planes of one frame with a published row count: rows
//! below it are final and readable by any thread, rows at or above it
//! belong to the single [`RefPlanesWriter`] of the decoding frame.
//!
//! Soundness: a reader only ever forms a slice over the published
//! prefix (loaded with `Acquire` after the writer's `Release` store),
//! the writer only ever forms a slice over unpublished rows (checked
//! against its own published count, which only it advances), and a
//! published row is never written again — so no sample is written
//! while a shared slice covers it.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::Error;

/// `published` once every row of every plane is final.
const COMPLETE: u32 = u32::MAX;
/// `published` once the writer was dropped before completing (the
/// frame's decode failed): readers error instead of waiting forever.
const FAILED: u32 = u32::MAX - 1;

/// The stored planes of one frame, readable by published prefix.
pub(crate) struct RefPlanes {
    bufs: Vec<Box<[UnsafeCell<u16>]>>,
    /// `(width, height)` per plane; the width is the row stride.
    dims: Vec<(u32, u32)>,
    /// Chroma vertical subsampling (`subsampling_y`).
    sub_y: u32,
    /// Luma rows published (rows of plane `p` published:
    /// `published >> subY(p)`), or [`COMPLETE`] / [`FAILED`].
    published: AtomicU32,
    lock: Mutex<()>,
    cond: Condvar,
}

// SAFETY: the `UnsafeCell` samples are shared under the published-prefix
// protocol stated in the module docs — shared slices cover only
// published rows, the unique writer's mutable slices only unpublished
// ones, and the hand-over is an `Acquire` / `Release` pair on
// `published`.
unsafe impl Sync for RefPlanes {}

impl std::fmt::Debug for RefPlanes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefPlanes")
            .field("dims", &self.dims)
            .field("published", &self.published.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

fn cells(len: usize) -> Box<[UnsafeCell<u16>]> {
    (0..len).map(|_| UnsafeCell::new(0)).collect()
}

impl RefPlanes {
    /// Already-final planes (the serial decode's store).
    pub(crate) fn complete(planes: Vec<Vec<u16>>, dims: Vec<(u32, u32)>, sub_y: u32) -> Arc<Self> {
        let bufs = planes
            .into_iter()
            .map(|p| p.into_iter().map(UnsafeCell::new).collect())
            .collect();
        Arc::new(Self {
            bufs,
            dims,
            sub_y,
            published: AtomicU32::new(COMPLETE),
            lock: Mutex::new(()),
            cond: Condvar::new(),
        })
    }

    /// Empty planes of `dims` and their unique writer.
    pub(crate) fn with_writer(dims: Vec<(u32, u32)>, sub_y: u32) -> (Arc<Self>, RefPlanesWriter) {
        let bufs = dims
            .iter()
            .map(|&(w, h)| cells((w as usize) * (h as usize)))
            .collect();
        let planes = Arc::new(Self {
            bufs,
            dims,
            sub_y,
            published: AtomicU32::new(0),
            lock: Mutex::new(()),
            cond: Condvar::new(),
        });
        let writer = RefPlanesWriter {
            planes: Arc::clone(&planes),
            published: 0,
        };
        (planes, writer)
    }

    /// `(width, height)` per plane.
    pub(crate) fn dims(&self) -> &[(u32, u32)] {
        &self.dims
    }

    /// Luma height (rows of plane 0).
    pub(crate) fn luma_height(&self) -> u32 {
        self.dims.first().map_or(0, |d| d.1)
    }

    fn plane_rows(&self, plane: usize, published: u32) -> usize {
        let h = self.dims[plane].1 as usize;
        match published {
            COMPLETE => h,
            FAILED => 0,
            p => {
                let sy = if plane == 0 { 0 } else { self.sub_y };
                ((p >> sy) as usize).min(h)
            }
        }
    }

    /// `true` once every row is final.
    pub(crate) fn is_complete(&self) -> bool {
        self.published.load(Ordering::Acquire) == COMPLETE
    }

    /// `true` once every row is final or the producing decode failed.
    pub(crate) fn is_settled(&self) -> bool {
        matches!(self.published.load(Ordering::Acquire), COMPLETE | FAILED)
    }

    /// The readable prefix of `plane` (the rows published so far).
    pub(crate) fn readable(&self, plane: usize) -> &[u16] {
        let p = self.published.load(Ordering::Acquire);
        let len = self.plane_rows(plane, p) * self.dims[plane].0 as usize;
        let buf = &self.bufs[plane];
        // SAFETY: the first `len` samples are published rows, which the
        // writer never touches again (module docs); `UnsafeCell<u16>`
        // has the layout of `u16`.
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u16>(), len) }
    }

    /// The whole of `plane` once complete.
    pub(crate) fn full(&self, plane: usize) -> Option<&[u16]> {
        self.is_complete().then(|| self.readable(plane))
    }

    /// Block until at least `luma_rows` luma rows (and the matching
    /// chroma rows) are published, or every row when `luma_rows`
    /// reaches the frame height.
    ///
    /// ## Errors
    ///
    /// The producing frame's decode failed.
    pub(crate) fn wait_rows(&self, luma_rows: u32) -> Result<(), Error> {
        let need = if luma_rows >= self.luma_height() {
            COMPLETE
        } else {
            luma_rows
        };
        let ready = |p: u32| p == COMPLETE || (p != FAILED && need != COMPLETE && p >= need);
        let p = self.published.load(Ordering::Acquire);
        if ready(p) {
            return Ok(());
        }
        let mut g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let p = self.published.load(Ordering::Acquire);
            if ready(p) {
                return Ok(());
            }
            if p == FAILED {
                return Err(Error::PartitionWalkOutOfRange);
            }
            g = self.cond.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Block until every row is final.
    ///
    /// ## Errors
    ///
    /// The producing frame's decode failed.
    pub(crate) fn wait_complete(&self) -> Result<(), Error> {
        self.wait_rows(u32::MAX)
    }

    fn set_published(&self, p: u32) {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.published.store(p, Ordering::Release);
        self.cond.notify_all();
    }
}

/// The unique writer of a [`RefPlanes`]: writes unpublished rows and
/// publishes them top-down. Dropping it before [`Self::finish`] marks
/// the planes failed.
pub(crate) struct RefPlanesWriter {
    planes: Arc<RefPlanes>,
    /// Luma rows published by this writer (mirrors `planes.published`).
    published: u32,
}

impl std::fmt::Debug for RefPlanesWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefPlanesWriter")
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

impl RefPlanesWriter {
    /// The planes this writer fills.
    pub(crate) fn planes(&self) -> &Arc<RefPlanes> {
        &self.planes
    }

    /// Plane rows `[ y0, y1 )` of `plane`, which must lie at or below
    /// the published prefix (panics otherwise — a published row is
    /// never written again).
    pub(crate) fn rows_mut(&mut self, plane: usize, y0: usize, y1: usize) -> &mut [u16] {
        let first_unpublished = self.planes.plane_rows(plane, self.published);
        let (w, h) = self.planes.dims[plane];
        assert!(
            y0 >= first_unpublished && y0 <= y1 && y1 <= h as usize,
            "reference rows {y0}..{y1} of plane {plane} are published or out of range"
        );
        let w = w as usize;
        let buf = &self.planes.bufs[plane];
        // SAFETY: rows `y0..y1` are unpublished, so no reader slice
        // covers them, and `&mut self` keeps this the only live
        // mutable slice of the planes.
        unsafe {
            std::slice::from_raw_parts_mut(
                UnsafeCell::raw_get(buf.as_ptr().add(y0 * w)),
                (y1 - y0) * w,
            )
        }
    }

    /// Publish the first `luma_rows` luma rows (and the matching chroma
    /// rows); `luma_rows >=` the frame height publishes everything.
    pub(crate) fn publish(&mut self, luma_rows: u32) {
        if luma_rows >= self.planes.luma_height() {
            self.published = COMPLETE;
            self.planes.set_published(COMPLETE);
        } else if luma_rows > self.published {
            self.published = luma_rows;
            self.planes.set_published(luma_rows);
        }
    }

    /// Publish every row.
    pub(crate) fn finish(mut self) {
        self.publish(u32::MAX);
    }
}

impl Drop for RefPlanesWriter {
    fn drop(&mut self) {
        if self.published != COMPLETE {
            self.planes.set_published(FAILED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_see_published_prefix_only() {
        let (planes, mut w) = RefPlanes::with_writer(vec![(4, 8), (2, 4), (2, 4)], 1);
        assert_eq!(planes.readable(0).len(), 0);
        w.rows_mut(0, 0, 4).fill(7);
        w.rows_mut(1, 0, 2).fill(9);
        w.publish(4);
        assert_eq!(planes.readable(0), &[7u16; 16][..]);
        assert_eq!(planes.readable(1), &[9u16; 4][..]);
        assert!(planes.full(0).is_none());
        w.rows_mut(0, 4, 8).fill(1);
        w.finish();
        assert_eq!(planes.full(0).map(<[u16]>::len), Some(32));
        assert_eq!(planes.readable(2).len(), 8);
    }

    #[test]
    #[should_panic(expected = "published or out of range")]
    fn published_rows_are_never_rewritten() {
        let (_planes, mut w) = RefPlanes::with_writer(vec![(4, 8)], 0);
        w.publish(4);
        let _ = w.rows_mut(0, 2, 6);
    }

    #[test]
    fn dropped_writer_fails_waiters() {
        let (planes, w) = RefPlanes::with_writer(vec![(4, 8)], 0);
        let reader = {
            let planes = Arc::clone(&planes);
            std::thread::spawn(move || planes.wait_rows(6))
        };
        drop(w);
        assert!(reader.join().unwrap().is_err());
        assert!(planes.wait_complete().is_err());
    }

    #[test]
    fn threaded_rows_arrive_in_order() {
        let (planes, mut w) = RefPlanes::with_writer(vec![(8, 64), (4, 32), (4, 32)], 1);
        let readers: Vec<_> = (0..3)
            .map(|k| {
                let planes = Arc::clone(&planes);
                std::thread::spawn(move || {
                    for need in (8..=64).step_by(8) {
                        planes.wait_rows(need).unwrap();
                        let rows = planes.readable(0);
                        assert!(rows.len() >= (need as usize).min(64) * 8, "reader {k}");
                        for (i, &v) in rows.iter().enumerate() {
                            assert_eq!(v, (i / 8) as u16);
                        }
                        let c = planes.readable(1);
                        assert!(c.len() >= (need as usize / 2) * 4);
                    }
                })
            })
            .collect();
        for band in 0..8usize {
            for (y, row) in w
                .rows_mut(0, band * 8, band * 8 + 8)
                .chunks_mut(8)
                .enumerate()
            {
                row.fill((band * 8 + y) as u16);
            }
            w.rows_mut(1, band * 4, band * 4 + 4).fill(band as u16);
            w.rows_mut(2, band * 4, band * 4 + 4).fill(band as u16);
            w.publish(((band + 1) * 8) as u32);
        }
        w.finish();
        for r in readers {
            r.join().unwrap();
        }
    }
}
