//! Piecewise-linear scaling LUT for film grain intensity shaping
//! (spec §7.20.3.3).
//!
//! AV1 transmits up to 14 `(value, scale)` control points per plane;
//! decoders expand them into a 256-entry LUT by piecewise-linear
//! interpolation between consecutive points. Values below the first
//! point clamp to that point's scale; values above the last point
//! clamp to the last point's scale.

/// 256-entry scaling LUT.
#[derive(Clone, Copy, Debug)]
pub struct ScalingLut(pub [u8; 256]);

impl Default for ScalingLut {
    fn default() -> Self {
        Self([0u8; 256])
    }
}

/// One `(value, scale)` control point from `film_grain_params`.
/// `value` is the luma level at which `scale` applies (0..=255 for
/// 8-bit, shifted for HBD); `scale` is the multiplier.
#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub value: u8,
    pub scale: u8,
}

/// Expand a sorted list of control points into a full 256-entry LUT.
/// `points` must be sorted by `value` ascending; the function returns
/// a zero LUT when there are no points.
///
/// Written for 8-bit samples. For 10/12-bit the spec indexes the LUT
/// with `(sample >> (bit_depth - 8))` — see [`ScalingLut::lookup_hbd`].
pub fn build_lut(points: &[Point]) -> ScalingLut {
    let mut lut = ScalingLut([0u8; 256]);
    if points.is_empty() {
        return lut;
    }
    // Values below the first point clamp to its scale.
    let first = points[0];
    for i in 0..first.value as usize {
        lut.0[i] = first.scale;
    }
    // Piecewise-linear between consecutive points.
    for p in 0..points.len() - 1 {
        let lo = points[p];
        let hi = points[p + 1];
        if hi.value <= lo.value {
            continue;
        }
        let span = hi.value as i32 - lo.value as i32;
        for i in lo.value as i32..=hi.value as i32 {
            let d = i - lo.value as i32;
            let scale = lo.scale as i32 * (span - d) + hi.scale as i32 * d;
            lut.0[i as usize] = ((scale + span / 2) / span) as u8;
        }
    }
    // Values above the last point clamp to its scale.
    let last = points[points.len() - 1];
    for i in last.value as usize + 1..256 {
        lut.0[i] = last.scale;
    }
    // Exact-value entries: the piecewise-linear loop covers
    // multi-point boundaries, but with a single control point the
    // exact value slot (e.g. lut[128] when value=128) is never
    // written. Set it explicitly here.
    for p in points {
        lut.0[p.value as usize] = p.scale;
    }
    lut
}

impl ScalingLut {
    /// Lookup for an 8-bit sample.
    pub fn lookup(&self, sample: u8) -> u8 {
        self.0[sample as usize]
    }

    /// Lookup for a 10/12-bit sample. `bit_depth` must be 8, 10 or 12.
    pub fn lookup_hbd(&self, sample: u16, bit_depth: u32) -> u8 {
        let shift = bit_depth.saturating_sub(8);
        self.0[(sample >> shift) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_below_first_point() {
        let lut = build_lut(&[
            Point {
                value: 32,
                scale: 40,
            },
            Point {
                value: 200,
                scale: 10,
            },
        ]);
        for i in 0..32 {
            assert_eq!(lut.0[i], 40);
        }
    }

    #[test]
    fn clamps_above_last_point() {
        let lut = build_lut(&[
            Point {
                value: 32,
                scale: 40,
            },
            Point {
                value: 200,
                scale: 10,
            },
        ]);
        for i in 201..256 {
            assert_eq!(lut.0[i], 10);
        }
    }

    #[test]
    fn interpolates_linearly() {
        let lut = build_lut(&[
            Point { value: 0, scale: 0 },
            Point {
                value: 200,
                scale: 200,
            },
        ]);
        assert!(lut.0[100] >= 95 && lut.0[100] <= 105);
        assert_eq!(lut.0[0], 0);
        assert_eq!(lut.0[200], 200);
    }

    #[test]
    fn empty_is_zero() {
        let lut = build_lut(&[]);
        for i in 0..256 {
            assert_eq!(lut.0[i], 0);
        }
    }
}
