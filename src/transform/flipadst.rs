//! Flipped-ADST inverse transforms — §7.7.2.2.
//!
//! Flipped-ADST is the same as inverse ADST followed by a reversal of
//! the spatial-domain output: the forward direction reverses its
//! output order, so the inverse mirrors it.
//!
//! The 4/8/16-point variants live alongside `iadst4/8/16` (see
//! [`crate::transform::adst4`] etc.) — this file carries only the
//! utility reversal used by the 2D dispatcher for mixed V/H variants.

/// Reverse a 1-D slice in place. Used after `iadst*` to produce
/// flipped-ADST output when the spec's `TxType` names the flip
/// explicitly, and by the V/H mixed variants before/after the ADST
/// kernel.
#[inline]
pub fn flip_1d(buf: &mut [i32]) {
    let n = buf.len();
    for i in 0..n / 2 {
        buf.swap(i, n - 1 - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_1d_reverses_even_and_odd_lengths() {
        let mut a = [1, 2, 3, 4];
        flip_1d(&mut a);
        assert_eq!(a, [4, 3, 2, 1]);

        let mut b = [1, 2, 3, 4, 5];
        flip_1d(&mut b);
        assert_eq!(b, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn flip_1d_noop_for_len_0_and_1() {
        let mut a: [i32; 0] = [];
        flip_1d(&mut a);
        let mut b = [42];
        flip_1d(&mut b);
        assert_eq!(b, [42]);
    }
}
