//! CDEF direction search — §7.15.3.2.
//!
//! Ported from goavif `av1/cdef/direction.go` + `direction16.go`.

/// `(row_offset, col_offset)` vectors for the 8 CDEF directions. Each
/// direction has two "distances": distance 1 (immediate neighbours
/// along the line) and distance 2 (one sample further along).
pub const DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]], // 0 — shallow up-right / down-left
    [[0, 1], [-1, 2]],  // 1
    [[0, 1], [0, 2]],   // 2 — horizontal
    [[0, 1], [1, 2]],   // 3
    [[1, 1], [2, 2]],   // 4 — diagonal
    [[1, 0], [2, 1]],   // 5
    [[1, 0], [2, 0]],   // 6 — vertical
    [[1, 0], [2, -1]],  // 7
];

/// Division LUT used by [`find_direction`] — matches libaom's
/// `div_table` ({0, 840/1, 840/2, ..., 840/8}).
pub const DIV_TABLE_FIND: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// 8-way direction search on an 8×8 block (libaom `cdef_find_dir_c`).
/// Returns `(direction, variance_diff)` where the variance diff is
/// between the chosen direction's cost and its orthogonal sibling.
#[allow(clippy::needless_range_loop)]
pub fn find_direction(src: &[u8], stride: usize, x: usize, y: usize) -> (usize, i32) {
    let mut partial = [[0i32; 15]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let xv = (src[(y + i) * stride + (x + j)] as i32) - 128;
            partial[0][i + j] += xv;
            partial[1][i + j / 2] += xv;
            partial[2][i] += xv;
            partial[3][3 + i - j / 2] += xv;
            partial[4][7 + i - j] += xv;
            partial[5][3 - i / 2 + j] += xv;
            partial[6][j] += xv;
            partial[7][i / 2 + j] += xv;
        }
    }

    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE_FIND[8];
    cost[6] *= DIV_TABLE_FIND[8];

    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE_FIND[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE_FIND[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE_FIND[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE_FIND[8];

    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE_FIND[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE_FIND[2 * j + 2];
        }
    }

    let mut best_cost = cost[0];
    let mut dir = 0usize;
    for i in 1..8 {
        if cost[i] > best_cost {
            best_cost = cost[i];
            dir = i;
        }
    }
    let ortho_cost = cost[dir ^ 4];
    (dir, best_cost - ortho_cost)
}

/// 16-bit counterpart of [`find_direction`]. Centres samples around
/// the bit-depth midpoint then right-shifts by `bit_depth - 8` so the
/// squared partial sums match the 8-bit path's dynamic range.
#[allow(clippy::needless_range_loop)]
pub fn find_direction16(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    bit_depth: u32,
) -> (usize, i32) {
    let mut partial = [[0i32; 15]; 8];
    let mid = 1i32 << (bit_depth - 1);
    let shift = bit_depth - 8;
    for i in 0..8 {
        for j in 0..8 {
            let raw = src[(y + i) * stride + (x + j)] as i32;
            let xv = (raw - mid) >> shift;
            partial[0][i + j] += xv;
            partial[1][i + j / 2] += xv;
            partial[2][i] += xv;
            partial[3][3 + i - j / 2] += xv;
            partial[4][7 + i - j] += xv;
            partial[5][3 - i / 2 + j] += xv;
            partial[6][j] += xv;
            partial[7][i / 2 + j] += xv;
        }
    }

    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE_FIND[8];
    cost[6] *= DIV_TABLE_FIND[8];

    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE_FIND[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE_FIND[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE_FIND[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE_FIND[8];

    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE_FIND[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE_FIND[2 * j + 2];
        }
    }

    let mut best_cost = cost[0];
    let mut dir = 0usize;
    for i in 1..8 {
        if cost[i] > best_cost {
            best_cost = cost[i];
            dir = i;
        }
    }
    let ortho_cost = cost[dir ^ 4];
    (dir, best_cost - ortho_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_direction_flat_block_returns_zero_variance() {
        let src = vec![100u8; 16 * 16];
        let (_, var) = find_direction(&src, 16, 4, 4);
        assert_eq!(var, 0);
    }

    #[test]
    fn find_direction_detects_horizontal_line() {
        let mut src = vec![100u8; 16 * 16];
        // Horizontal streak one sample bright across row 5 of the 8x8
        // block located at (4, 4).
        for c in 0..8 {
            src[9 * 16 + (4 + c)] = 200;
        }
        let (dir, var) = find_direction(&src, 16, 4, 4);
        assert!(var > 0, "expected nonzero variance for a line");
        // The horizontal direction is 2 in the 8-way ring.
        assert_eq!(dir, 2, "expected horizontal dir (2), got {dir}");
    }
}
