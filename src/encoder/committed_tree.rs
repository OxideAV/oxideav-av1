//! r479 — the committed superblock trees in COMPACT form.
//!
//! The intra drivers retain every superblock's committed
//! [`SyntaxNode`] tree after the search: the §5.9.19 CDEF and §5.9.20
//! loop-restoration elections re-emit the tile from them (with
//! `cdef_idx` literals stamped / the §5.11.57 `write_lr` interleave).
//! A boxed [`SyntaxBlock`] is 264 bytes plus a `Vec` per transform
//! unit plus 8 bytes per non-zero coefficient — 47 MiB of trees for
//! a 12 MP photograph at `base_q_idx = 102`. [`CommittedTree`] keeps
//! the same information in ~16 MiB: a pre-order shape string, a
//! 32-byte record per leaf, one arena of transform-unit headers and
//! one of `(raster index, value)` pairs at 4 bytes each (values
//! outside `i16` escape to a side table), and a boxed [`SyntaxBlock`]
//! only for the leaves the packed record cannot carry (palette,
//! intra-block-copy, inter, variable-transform or delta-lf
//! commitments). [`CommittedTree::expand`] rebuilds the exact
//! [`SyntaxNode`] tree for the writer, one superblock at a time.

use crate::cdf::FRAME_LF_COUNT;
use crate::encoder::partition_tree::{
    SyntaxBlock, SyntaxNode, SyntaxPalette, TuQuant, TuQuantList,
};

const TAG_LEAF: u8 = 0;
const TAG_SPLIT: u8 = 1;
const TAG_HORZ: u8 = 2;
const TAG_VERT: u8 = 3;
const TAG_HORZ_A: u8 = 4;
const TAG_HORZ_B: u8 = 5;
const TAG_VERT_A: u8 = 6;
const TAG_VERT_B: u8 = 7;
const TAG_HORZ_4: u8 = 8;
const TAG_VERT_4: u8 = 9;

/// `Option<u8>` / `Option<i8>` as one byte.
const NONE_U8: u8 = 0xFF;
const NONE_I8: i8 = i8::MIN;

/// Escape marker on the raster-index half of a coefficient pair: the
/// value half is then an index into [`CommittedTree::escapes`].
const ESCAPE: u16 = 0x8000;

/// One packed intra leaf (see the module docs).
#[derive(Clone, Copy)]
struct PackedLeaf {
    /// Index into [`CommittedTree::full`] when the leaf could not be
    /// packed, else `u32::MAX`.
    full: u32,
    reduced_delta_q_index: i32,
    /// First transform-unit header in [`CommittedTree::tus`].
    tu_start: u32,
    tu_count: u16,
    /// Range of [`CommittedTree::tx_types`].
    tx_start: u32,
    tx_count: u16,
    skip: u8,
    segment_id: u8,
    cdef_idx: i8,
    y_mode: u8,
    uv_mode: u8,
    angle_delta_y: i8,
    angle_delta_uv: i8,
    cfl_alpha_u: i8,
    cfl_alpha_v: i8,
    use_filter_intra: u8,
    filter_intra_mode: u8,
    tx_size: u8,
}

/// One transform unit: `len` coefficients, `nz_count` non-zero pairs
/// from `nz_start` in [`CommittedTree::nz`].
#[derive(Clone, Copy)]
struct PackedTu {
    nz_start: u32,
    len: u16,
    nz_count: u16,
}

/// A superblock's committed tree in compact form.
#[derive(Clone, Default)]
pub(crate) struct CommittedTree {
    shape: Vec<u8>,
    leaves: Vec<PackedLeaf>,
    tus: Vec<PackedTu>,
    nz: Vec<(u16, i16)>,
    escapes: Vec<i32>,
    tx_types: Vec<u8>,
    full: Vec<SyntaxBlock>,
}

impl CommittedTree {
    /// Pack `node`.
    pub(crate) fn pack(node: &SyntaxNode) -> Self {
        let mut t = Self::default();
        t.push_node(node);
        t.shape.shrink_to_fit();
        t.leaves.shrink_to_fit();
        t.tus.shrink_to_fit();
        t.nz.shrink_to_fit();
        t.escapes.shrink_to_fit();
        t.tx_types.shrink_to_fit();
        t.full.shrink_to_fit();
        t
    }

    /// r429 — stamp the §5.11.56 `cdef_idx` onto every leaf (see
    /// [`SyntaxNode::stamp_cdef_idx`]).
    pub(crate) fn stamp_cdef_idx(&mut self, idx: i8) {
        for l in &mut self.leaves {
            l.cdef_idx = idx;
        }
        for b in &mut self.full {
            b.cdef_idx = idx;
        }
    }

    fn push_node(&mut self, node: &SyntaxNode) {
        match node {
            SyntaxNode::Leaf(b) => {
                self.shape.push(TAG_LEAF);
                self.push_leaf(b);
            }
            SyntaxNode::Split(q) => {
                self.shape.push(TAG_SPLIT);
                for n in q.iter() {
                    self.push_node(n);
                }
            }
            SyntaxNode::Horz(b) | SyntaxNode::Vert(b) => {
                self.shape.push(if matches!(node, SyntaxNode::Horz(_)) {
                    TAG_HORZ
                } else {
                    TAG_VERT
                });
                for blk in b.iter() {
                    self.push_leaf(blk);
                }
            }
            SyntaxNode::HorzA(b)
            | SyntaxNode::HorzB(b)
            | SyntaxNode::VertA(b)
            | SyntaxNode::VertB(b) => {
                self.shape.push(match node {
                    SyntaxNode::HorzA(_) => TAG_HORZ_A,
                    SyntaxNode::HorzB(_) => TAG_HORZ_B,
                    SyntaxNode::VertA(_) => TAG_VERT_A,
                    _ => TAG_VERT_B,
                });
                for blk in b.iter() {
                    self.push_leaf(blk);
                }
            }
            SyntaxNode::Horz4(b) | SyntaxNode::Vert4(b) => {
                self.shape.push(if matches!(node, SyntaxNode::Horz4(_)) {
                    TAG_HORZ_4
                } else {
                    TAG_VERT_4
                });
                for blk in b.iter() {
                    self.push_leaf(blk);
                }
            }
        }
    }

    fn push_leaf(&mut self, b: &SyntaxBlock) {
        let packable = b.intrabc_mv.is_none()
            && b.inter.is_none()
            && b.var_tx_trees.is_empty()
            && b.reduced_delta_lf == [0; FRAME_LF_COUNT]
            && palette_is_default(&b.palette)
            && b.residual_quant.len() <= usize::from(u16::MAX)
            && b.residual_tx_type.len() <= usize::from(u16::MAX)
            && b.residual_quant
                .iter()
                .all(|t| t.len() <= usize::from(u16::MAX))
            && b.residual_quant
                .iter()
                .all(|t| t.nonzero_count() <= usize::from(u16::MAX));
        if !packable {
            self.full.push(b.clone());
            self.leaves.push(PackedLeaf {
                full: (self.full.len() - 1) as u32,
                reduced_delta_q_index: 0,
                tu_start: 0,
                tu_count: 0,
                tx_start: 0,
                tx_count: 0,
                skip: 0,
                segment_id: 0,
                cdef_idx: b.cdef_idx,
                y_mode: 0,
                uv_mode: NONE_U8,
                angle_delta_y: 0,
                angle_delta_uv: 0,
                cfl_alpha_u: NONE_I8,
                cfl_alpha_v: NONE_I8,
                use_filter_intra: 0,
                filter_intra_mode: NONE_U8,
                tx_size: NONE_U8,
            });
            return;
        }
        let tu_start = self.tus.len() as u32;
        for tu in b.residual_quant.iter() {
            let nz_start = self.nz.len() as u32;
            for &(i, q) in tu.nonzeros() {
                debug_assert!(i < ESCAPE);
                if let Ok(v) = i16::try_from(q) {
                    self.nz.push((i, v));
                } else {
                    let e = self.escapes.len();
                    debug_assert!(e <= usize::from(u16::MAX));
                    self.escapes.push(q);
                    self.nz.push((i | ESCAPE, e as i16));
                }
            }
            self.tus.push(PackedTu {
                nz_start,
                len: tu.len() as u16,
                nz_count: tu.nonzero_count() as u16,
            });
        }
        let tx_start = self.tx_types.len() as u32;
        self.tx_types.extend_from_slice(&b.residual_tx_type);
        self.leaves.push(PackedLeaf {
            full: u32::MAX,
            reduced_delta_q_index: b.reduced_delta_q_index,
            tu_start,
            tu_count: b.residual_quant.len() as u16,
            tx_start,
            tx_count: b.residual_tx_type.len() as u16,
            skip: b.skip,
            segment_id: b.segment_id,
            cdef_idx: b.cdef_idx,
            y_mode: b.y_mode,
            uv_mode: b.uv_mode.unwrap_or(NONE_U8),
            angle_delta_y: b.angle_delta_y,
            angle_delta_uv: b.angle_delta_uv,
            cfl_alpha_u: b.cfl_alpha_u.unwrap_or(NONE_I8),
            cfl_alpha_v: b.cfl_alpha_v.unwrap_or(NONE_I8),
            use_filter_intra: b.use_filter_intra,
            filter_intra_mode: b.filter_intra_mode.unwrap_or(NONE_U8),
            tx_size: b.tx_size.unwrap_or(NONE_U8),
        });
    }

    /// Rebuild the [`SyntaxNode`] tree.
    pub(crate) fn expand(&self) -> SyntaxNode {
        let mut pos = (0usize, 0usize);
        let node = self.expand_node(&mut pos);
        debug_assert_eq!(pos.0, self.shape.len());
        debug_assert_eq!(pos.1, self.leaves.len());
        node
    }

    fn expand_node(&self, pos: &mut (usize, usize)) -> SyntaxNode {
        let tag = self.shape[pos.0];
        pos.0 += 1;
        let leaf = |pos: &mut (usize, usize)| {
            let b = self.expand_leaf(pos.1);
            pos.1 += 1;
            b
        };
        match tag {
            TAG_LEAF => SyntaxNode::Leaf(leaf(pos)),
            TAG_SPLIT => {
                let a = self.expand_node(pos);
                let b = self.expand_node(pos);
                let c = self.expand_node(pos);
                let d = self.expand_node(pos);
                SyntaxNode::Split([Box::new(a), Box::new(b), Box::new(c), Box::new(d)])
            }
            TAG_HORZ | TAG_VERT => {
                let pair = [leaf(pos), leaf(pos)];
                if tag == TAG_HORZ {
                    SyntaxNode::Horz(pair)
                } else {
                    SyntaxNode::Vert(pair)
                }
            }
            TAG_HORZ_A | TAG_HORZ_B | TAG_VERT_A | TAG_VERT_B => {
                let trio = [leaf(pos), leaf(pos), leaf(pos)];
                match tag {
                    TAG_HORZ_A => SyntaxNode::HorzA(trio),
                    TAG_HORZ_B => SyntaxNode::HorzB(trio),
                    TAG_VERT_A => SyntaxNode::VertA(trio),
                    _ => SyntaxNode::VertB(trio),
                }
            }
            _ => {
                let quad = [leaf(pos), leaf(pos), leaf(pos), leaf(pos)];
                if tag == TAG_HORZ_4 {
                    SyntaxNode::Horz4(quad)
                } else {
                    SyntaxNode::Vert4(quad)
                }
            }
        }
    }

    fn expand_leaf(&self, i: usize) -> Box<SyntaxBlock> {
        let l = &self.leaves[i];
        if l.full != u32::MAX {
            let mut b = Box::new(self.full[l.full as usize].clone());
            b.cdef_idx = l.cdef_idx;
            return b;
        }
        let tus = &self.tus[l.tu_start as usize..l.tu_start as usize + usize::from(l.tu_count)];
        let quant: Vec<TuQuant> = tus
            .iter()
            .map(|tu| {
                let pairs =
                    &self.nz[tu.nz_start as usize..tu.nz_start as usize + usize::from(tu.nz_count)];
                let nz: Vec<(u16, i32)> = pairs
                    .iter()
                    .map(|&(i, v)| {
                        if i & ESCAPE != 0 {
                            (i & !ESCAPE, self.escapes[v as u16 as usize])
                        } else {
                            (i, i32::from(v))
                        }
                    })
                    .collect();
                TuQuant::from_sparse(u32::from(tu.len), nz)
            })
            .collect();
        let opt_u8 = |v: u8| (v != NONE_U8).then_some(v);
        let opt_i8 = |v: i8| (v != NONE_I8).then_some(v);
        Box::new(SyntaxBlock {
            skip: l.skip,
            segment_id: l.segment_id,
            cdef_idx: l.cdef_idx,
            reduced_delta_q_index: l.reduced_delta_q_index,
            reduced_delta_lf: [0; FRAME_LF_COUNT],
            intrabc_mv: None,
            y_mode: l.y_mode,
            uv_mode: opt_u8(l.uv_mode),
            angle_delta_y: l.angle_delta_y,
            angle_delta_uv: l.angle_delta_uv,
            cfl_alpha_u: opt_i8(l.cfl_alpha_u),
            cfl_alpha_v: opt_i8(l.cfl_alpha_v),
            use_filter_intra: l.use_filter_intra,
            filter_intra_mode: opt_u8(l.filter_intra_mode),
            palette: SyntaxPalette::default(),
            residual_quant: TuQuantList(quant),
            tx_size: opt_u8(l.tx_size),
            residual_tx_type: self.tx_types
                [l.tx_start as usize..l.tx_start as usize + usize::from(l.tx_count)]
                .to_vec(),
            var_tx_trees: Vec::new(),
            inter: None,
        })
    }
}

fn palette_is_default(p: &SyntaxPalette) -> bool {
    p.size_y == 0
        && p.size_uv == 0
        && !p.delta_encode_v
        && p.color_map_y.is_empty()
        && p.color_map_uv.is_empty()
        && p.colors_y.iter().all(|&c| c == 0)
        && p.colors_u.iter().all(|&c| c == 0)
        && p.colors_v.iter().all(|&c| c == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::partition_tree::SyntaxBlock;

    fn leaf(seed: u32, nz: &[(u16, i32)], len: usize) -> Box<SyntaxBlock> {
        let mut b = SyntaxBlock::skip_leaf((seed % 13) as u8, Some((seed % 14) as u8));
        b.skip = 0;
        b.segment_id = (seed % 8) as u8;
        b.cdef_idx = (seed % 8) as i8;
        b.reduced_delta_q_index = seed as i32 - 7;
        b.angle_delta_y = (seed % 7) as i8 - 3;
        b.angle_delta_uv = 3 - (seed % 7) as i8;
        b.cfl_alpha_u = if seed % 2 == 0 { Some(-5) } else { None };
        b.cfl_alpha_v = if seed % 3 == 0 { Some(12) } else { None };
        b.use_filter_intra = (seed % 2) as u8;
        b.filter_intra_mode = (seed % 2 == 1).then_some((seed % 5) as u8);
        b.tx_size = (seed % 4 != 0).then_some((seed % 19) as u8);
        b.residual_tx_type = vec![(seed % 16) as u8; (seed % 3) as usize];
        let mut dense = vec![0i32; len];
        for &(i, q) in nz {
            dense[i as usize] = q;
        }
        b.residual_quant = TuQuantList::from(vec![dense.clone(), dense]);
        Box::new(b)
    }

    fn same_block(a: &SyntaxBlock, b: &SyntaxBlock) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    fn same_tree(a: &SyntaxNode, b: &SyntaxNode) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn packed_trees_expand_to_the_same_syntax() {
        let big = leaf(5, &[(0, 40_000), (17, -70_000), (4095, 3)], 4096);
        let tree = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(leaf(1, &[(3, -1), (9, 77)], 256))),
            Box::new(SyntaxNode::Horz([leaf(2, &[], 64), leaf(3, &[(1, 1)], 64)])),
            Box::new(SyntaxNode::HorzA([
                leaf(4, &[(0, 2)], 16),
                big,
                leaf(6, &[(2, -32768), (3, 32767)], 16),
            ])),
            Box::new(SyntaxNode::Split([
                Box::new(SyntaxNode::Vert4([
                    leaf(7, &[], 16),
                    leaf(8, &[], 16),
                    leaf(9, &[(5, 5)], 16),
                    leaf(10, &[], 16),
                ])),
                Box::new(SyntaxNode::dummy_oob()),
                Box::new(SyntaxNode::VertB([
                    leaf(11, &[], 16),
                    leaf(12, &[], 16),
                    leaf(13, &[], 16),
                ])),
                Box::new(SyntaxNode::Leaf(Box::new(SyntaxBlock::skip_leaf(0, None)))),
            ])),
        ]);
        let packed = CommittedTree::pack(&tree);
        assert!(packed.full.is_empty());
        assert_eq!(packed.escapes.len(), 2 * 2);
        same_tree(&packed.expand(), &tree);
        let mut stamped = packed.clone();
        stamped.stamp_cdef_idx(3);
        let mut want = tree.clone();
        want.stamp_cdef_idx(3);
        same_tree(&stamped.expand(), &want);
    }

    #[test]
    fn unpackable_leaves_ride_a_full_block() {
        let mut b = leaf(1, &[(1, 1)], 16);
        b.palette.size_y = 2;
        b.palette.colors_y[0] = 10;
        b.palette.colors_y[1] = 20;
        b.palette.color_map_y = vec![0, 1, 1, 0];
        let mut c = leaf(2, &[], 16);
        c.intrabc_mv = Some([8, -16]);
        let tree = SyntaxNode::Vert([b, c]);
        let mut packed = CommittedTree::pack(&tree);
        assert_eq!(packed.full.len(), 2);
        same_tree(&packed.expand(), &tree);
        packed.stamp_cdef_idx(1);
        if let SyntaxNode::Vert([x, y]) = packed.expand() {
            assert_eq!(x.cdef_idx, 1);
            assert_eq!(y.cdef_idx, 1);
            let mut wx = tree_block(&tree, 0);
            wx.cdef_idx = 1;
            same_block(&x, &wx);
        } else {
            panic!("shape");
        }
    }

    fn tree_block(t: &SyntaxNode, i: usize) -> SyntaxBlock {
        t.asymmetric_blocks()[i].as_ref().clone()
    }
}
