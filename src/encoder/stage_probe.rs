//! r479 — encoder stage probe: a process-wide callback the intra
//! frame driver fires at every stage boundary (search, deblock
//! election, CDEF election, restoration election, assembly) with a
//! short stage label. Measurement harnesses (`examples/encode_bench`)
//! register a probe that samples their own allocator counters at
//! each boundary, which is how the README's per-stage memory table is
//! produced. Inert (one relaxed atomic load per boundary) when no
//! probe is registered; never on the per-block path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

type Probe = Box<dyn Fn(&'static str) + Send + Sync + 'static>;

static ARMED: AtomicBool = AtomicBool::new(false);
static PROBE: RwLock<Option<Probe>> = RwLock::new(None);

/// Register `probe` (or clear it with `None`). The callback runs on
/// the encoding thread at each stage boundary with the stage label.
#[doc(hidden)]
pub fn set_stage_probe(probe: Option<Probe>) {
    let mut g = PROBE.write().unwrap_or_else(|e| e.into_inner());
    ARMED.store(probe.is_some(), Ordering::Release);
    *g = probe;
}

/// Fire the probe with `label` when one is registered.
#[inline]
pub(crate) fn stage(label: &'static str) {
    if !ARMED.load(Ordering::Acquire) {
        return;
    }
    if let Ok(g) = PROBE.read() {
        if let Some(p) = g.as_ref() {
            p(label);
        }
    }
}
