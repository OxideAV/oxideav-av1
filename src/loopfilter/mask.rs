//! Deblocking threshold derivation — §7.14.3.
//!
//! Ported from goavif `av1/loopfilter/mask.go`.

use super::narrow::Thresholds;

/// Compute the deblocking `{limit, blimit, thresh}` triple from the
/// uncompressed frame header's `filter_level` + `sharpness`.
///
/// * `limit  = filter_level >> (sharpness >= 7 ? 2 : 1)`, min 1,
///   capped at `63 / (sharpness+1)` when `sharpness > 0`.
/// * `blimit = 2 * (filter_level + 2) + limit` (clamped to 255).
/// * `thresh = filter_level >> 4`.
pub fn derive_thresholds(filter_level: i32, sharpness: i32) -> Thresholds {
    let filter_level = filter_level.clamp(0, 63);
    let sharpness = sharpness.clamp(0, 7);

    let shift = if sharpness >= 7 { 2 } else { 1 };
    let mut limit = filter_level >> shift;

    if sharpness > 0 {
        let cap = 63 / (sharpness + 1);
        if limit > cap {
            limit = cap;
        }
    }
    if limit < 1 {
        limit = 1;
    }

    let mut blimit = 2 * (filter_level + 2) + limit;
    if blimit > 255 {
        blimit = 255;
    }
    let thresh = filter_level >> 4;

    Thresholds {
        limit: limit as u8,
        blimit: blimit as u8,
        thresh: thresh as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_for_mid_filter_level() {
        let th = derive_thresholds(32, 2);
        assert!(th.limit >= 1);
        assert!(th.blimit > 0);
    }

    #[test]
    fn zero_filter_level_gives_minimum_limit() {
        let th = derive_thresholds(0, 0);
        assert_eq!(th.limit, 1);
    }

    #[test]
    fn high_sharpness_caps_limit() {
        let th = derive_thresholds(60, 7);
        assert!(th.limit <= (63 / 8));
    }
}
