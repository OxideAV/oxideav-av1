//! Grain template generation (spec §7.20.2 / §7.20.3.5 / §7.20.3.6).
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/filmgrain/patch.go`
//! (MIT, KarpelesLab/goavif). The template is a pre-computed block
//! of grain samples shaped by the AR model: 73×73 for luma, 38×38 for
//! 4:2:0 chroma. Apply time looks up 32×32 sub-patches.

use super::ar::{apply_ar, generate_grain_template};

/// Rectangular block of pre-computed grain samples. The spec builds a
/// 73×73 luma template (§7.20.3.5); chroma uses 38×38 (§7.20.3.6).
#[derive(Clone, Debug)]
pub struct Template {
    pub data: Vec<i16>,
    pub rows: usize,
    pub cols: usize,
}

impl Default for Template {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }
}

/// Build the 73×73 luma template with LFSR samples optionally shaped
/// by an AR filter of the given `lag`. Pass `lag == 0` or mismatched
/// `coeffs` to skip shaping.
pub fn new_luma_template(seed: u16, lag: usize, coeffs: &[i8], shift: u32) -> Template {
    const ROWS: usize = 73;
    const COLS: usize = 73;
    let mut t = Template {
        data: generate_grain_template(COLS, ROWS, seed),
        rows: ROWS,
        cols: COLS,
    };
    apply_ar(&mut t.data, COLS, ROWS, lag, coeffs, shift);
    t
}

/// Build the 38×38 chroma template suitable for a 4:2:0 plane.
pub fn new_chroma_template(seed: u16, lag: usize, coeffs: &[i8], shift: u32) -> Template {
    const ROWS: usize = 38;
    const COLS: usize = 38;
    let mut t = Template {
        data: generate_grain_template(COLS, ROWS, seed),
        rows: ROWS,
        cols: COLS,
    };
    apply_ar(&mut t.data, COLS, ROWS, lag, coeffs, shift);
    t
}

impl Template {
    /// Return the template value at `(r, c)` with wrap-around
    /// indexing.
    pub fn sample(&self, r: i32, c: i32) -> i16 {
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }
        let rr = ((r % self.rows as i32) + self.rows as i32) % self.rows as i32;
        let cc = ((c % self.cols as i32) + self.cols as i32) % self.cols as i32;
        self.data[rr as usize * self.cols + cc as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_template_dimensions() {
        let t = new_luma_template(0x1234, 0, &[], 7);
        assert_eq!(t.rows, 73);
        assert_eq!(t.cols, 73);
        assert_eq!(t.data.len(), t.rows * t.cols);
    }

    #[test]
    fn chroma_template_dimensions() {
        let t = new_chroma_template(0x4567, 0, &[], 7);
        assert_eq!(t.rows, 38);
        assert_eq!(t.cols, 38);
    }

    #[test]
    fn sample_wraps() {
        let t = Template {
            data: vec![1, 2, 3, 4],
            rows: 2,
            cols: 2,
        };
        assert_eq!(t.sample(0, 0), 1);
        assert_eq!(t.sample(2, 0), 1);
        assert_eq!(t.sample(0, -1), 2);
    }
}
