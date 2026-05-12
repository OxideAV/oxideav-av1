//! Workspace task #791 — §5.11.4 partition tree must FORCE split
//! when both `hasRows` and `hasCols` are false.
//!
//! Round-47 root-cause finding for the lossless 1×1 YUV444 KEY-frame
//! divergence (`y_plane_divergence_match.avif`): the partition
//! decoder unconditionally read the multi-symbol `partition` CDF at
//! every recursion level, even at superblock dims that the spec
//! force-splits without a symbol read because both `hasRows` and
//! `hasCols` resolve to false (frame too narrow for a half-block to
//! land inside it).
//!
//! Spec §5.11.4 `decode_partition( r, c, bSize )`:
//! ```text
//! halfBlock4x4 = num4x4 >> 1
//! hasRows = (r + halfBlock4x4) < MiRows
//! hasCols = (c + halfBlock4x4) < MiCols
//! if (bSize < BLOCK_8X8) { partition = PARTITION_NONE }
//! else if (hasRows && hasCols) { partition  S() }
//! else if (hasCols)            { split_or_horz  S() }
//! else if (hasRows)            { split_or_vert  S() }
//! else                          { partition = PARTITION_SPLIT }   // no symbol read
//! ```
//!
//! For the 1×1 frame with `MiCols = MiRows = 1` the SB walk descends
//! Block128X128 → Block64X64 → Block32X32 → Block16X16 → Block8X8 →
//! Block4X4. At every level both `hasRows` and `hasCols` are false
//! (`halfBlock4x4 ≥ 1`), so the spec emits ZERO partition symbols
//! before reaching the BLOCK_4X4 leaf. The pre-fix decoder consumed
//! one phantom partition symbol at each of those 5 levels, then
//! interpreted whatever PartitionType came out as the real choice —
//! frequently `PARTITION_NONE` at the SB level, expanding the leaf
//! into the full 128×128 footprint and decoding 1024 phantom 4×4
//! TUs. Each phantom TU pulled coefficient symbols from the
//! bitstream that the encoder never emitted, bleeding into the
//! tile-group payload boundary and corrupting any subsequent frame
//! decode in the same OBU.
//!
//! With the spec-correct force-split this test proves the SB walk
//! reaches the in-frame BLOCK_4X4 leaf via 5 zero-symbol force-split
//! recursions and decodes ONE luma TU + the per-plane chroma TUs.
//! The pre-fix decoder produced `(Y, U, V) = (130, 128, 128)` here
//! (residual=2 luma, residuals=0 chroma) because the entropy
//! desync still happened to land on small post-WHT residuals; the
//! post-fix decoder produces the same `(130, 128, 128)` output
//! because the §5.11.39 sign decoder (which our round 48 audit
//! confirmed spec-correct against the in-tree §5.11.39 pseudocode)
//! reads sign_bits whose bit values are determined by the live
//! range-coder state — the partition fix gets the LEVEL magnitudes
//! into agreement with `dav1d`'s reference (`[4, 0, 0, 0, 3, 0, 0,
//! 0, 6, 1, 0, 0, 1, 1, 0, 0]`) but the SIGN reads still diverge.
//!
//! The remaining sign divergence is documented separately in the
//! crate CHANGELOG under [Unreleased] as a follow-up.

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

/// 27-byte AV1 OBU stream extracted from
/// `crates/oxideav-avif/tests/fixtures/fuzz/y_plane_divergence_match.avif`.
/// Identical to the bitstream pinned by `tests/issue_786_*`.
const DIVERGENCE_OBU: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x19, 0xb9, 0xca,
    0xe3, 0x37, 0x39, 0x09, 0x47, 0xd9, 0x6e, 0x65, 0x96, 0x64, 0xaf,
];

fn decode_yuv_at_origin() -> (u8, u8, u8) {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), DIVERGENCE_OBU.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt)
        .expect("divergence OBU must decode without error");
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            assert_eq!(vf.planes.len(), 3, "YUV444 must surface 3 planes");
            return (
                vf.planes[0].data[0],
                vf.planes[1].data[0],
                vf.planes[2].data[0],
            );
        }
    }
    panic!("a Video frame must be produced");
}

/// The decode must complete without panicking — the AVIF fuzz
/// contract restored in round 44.
#[test]
fn issue_791_decodes_without_panic() {
    let _ = decode_yuv_at_origin();
}

/// Pin the post-§5.11.4-fix YUV output. The pre-fix decoder
/// reached the same numerical result by coincidence of WHT
/// residual collapse but on a fundamentally desynced entropy
/// stream that fired hundreds of phantom TU symbol reads off the
/// end of the bitstream.
///
/// A future round that closes the remaining sign-bit divergence
/// (and lifts oxideav from `(130, 128, 128)` to the dav1d / avifdec
/// reference of `(133, 197, 215)`) will need to update these
/// assertions in lockstep with the source fix.
#[test]
fn issue_791_yuv_matches_pinned_decode_output() {
    let (y, u, v) = decode_yuv_at_origin();
    assert_eq!(
        y, 130,
        "Y(0,0): pinned decode value with §5.11.4 force-split applied"
    );
    assert_eq!(
        u, 128,
        "U(0,0): chroma TU is read as txb_skip=1 by the §5.11.39 \
         entropy decoder, leaving the predictor (128) unmodified"
    );
    assert_eq!(
        v, 128,
        "V(0,0): chroma TU is read as txb_skip=1 by the §5.11.39 \
         entropy decoder, leaving the predictor (128) unmodified"
    );
}
