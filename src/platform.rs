//! Target capabilities the decoder and encoder lean on.
//!
//! Two `std` facilities compile on every target but fail at run time
//! on some WebAssembly ones:
//!
//! * `std::time::Instant::now()` panics on `wasm32-unknown-unknown`
//!   (no clock), so the per-stage decode timings go through
//!   [`StageClock`], which reads zero there (#33);
//! * `std::thread::spawn` / `std::thread::scope` cannot start a thread
//!   on `wasm32-unknown-unknown` or on a WebAssembly target built
//!   without the `atomics` feature, so every worker budget goes
//!   through [`clamp_threads`], which pins it to one there (the serial
//!   path, byte-identical to every threaded one).

use std::time::Duration;

/// `false` on targets where `std` cannot start a thread.
pub(crate) const THREADS_AVAILABLE: bool = !cfg!(any(
    all(target_family = "wasm", not(target_feature = "atomics")),
    all(target_family = "wasm", target_os = "unknown"),
));

/// A worker budget the target can honour: `threads.max(1)` where
/// threads exist, `1` (serial) where they do not.
#[inline]
pub(crate) fn clamp_threads(threads: usize) -> usize {
    if THREADS_AVAILABLE { threads.max(1) } else { 1 }
}

/// A start mark for one decode-stage measurement: a monotonic
/// `Instant` where the target has a clock, nothing (every reading
/// zero) on `wasm32-unknown-unknown`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StageClock {
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    start: std::time::Instant,
}

impl StageClock {
    /// Mark "now".
    #[inline]
    pub(crate) fn now() -> Self {
        Self {
            #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
            start: std::time::Instant::now(),
        }
    }

    /// Wall time since [`Self::now`] (`Duration::ZERO` without a
    /// clock).
    #[inline]
    pub(crate) fn elapsed(&self) -> Duration {
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        {
            self.start.elapsed()
        }
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        {
            Duration::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_never_returns_zero() {
        assert_eq!(clamp_threads(0), 1);
        assert_eq!(clamp_threads(1), 1);
        assert_eq!(clamp_threads(8), if THREADS_AVAILABLE { 8 } else { 1 });
    }

    #[test]
    fn stage_clock_is_monotonic() {
        let c = StageClock::now();
        let a = c.elapsed();
        let b = c.elapsed();
        assert!(b >= a);
    }
}
