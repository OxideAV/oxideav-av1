//! r424 — screen-content polish witnesses (ladder item 5).
//!
//! Stream-level coverage for the §5.11.46 signed-delta V-plane arm
//! election through the PUBLIC key-frame entry point (the tree-level
//! election witness lives in the `key_frame` module tests), plus the
//! env-gated fixture-generation twin for the pinned corpus stream.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_key_frame_yuv420_with_q, Yuv420Frame};

/// 4-colour dither luma over a 2×2-cell chroma checker with a WIDE U
/// spread and a TIGHT V spread — UV-palette territory whose V entry
/// list (100, 106) codes cheaper through the §5.11.46
/// `delta_encode_palette_colors_v` chain than as direct literals.
fn vdelta_input() -> Yuv420Frame {
    const COLORS: [u8; 4] = [16, 80, 160, 240];
    let mut f = Yuv420Frame::filled(64, 64, 128);
    let mut state = 0x1234_5678u32 ^ 91;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for i in 0..64usize {
        for j in 0..64usize {
            f.y[i * 64 + j] = COLORS[(next() & 3) as usize];
        }
    }
    for i in 0..32usize {
        for j in 0..32usize {
            let cell = ((i / 2) + (j / 2)) & 1;
            f.u[i * 32 + j] = if cell == 0 { 64 } else { 192 };
            f.v[i * 32 + j] = if cell == 0 { 100 } else { 106 };
        }
    }
    f
}

/// The public key-frame path (its sequence header always selects
/// screen-content tools) must produce a conformant stream on the
/// tight-V-cluster content — the §5.11.46 delta-arm election is live
/// inside — and the spec driver must decode it byte-exact against
/// the encoder reconstruction.
#[test]
fn signed_delta_v_content_round_trips_via_public_api() {
    let input = vdelta_input();
    let enc = encode_key_frame_yuv420_with_q(&input, 60).expect("encode");
    let frames = decode_av1_spec(&enc.ivf_bytes).expect("spec driver");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].planes[0], enc.recon_y, "luma");
    assert_eq!(frames[0].planes[1], enc.recon_u, "U");
    assert_eq!(frames[0].planes[2], enc.recon_v, "V");
}

/// Env-gated fixture-generation twin (`OXIDEAV_AV1_SCC_FIXTURE_DIR`):
/// the signed-delta V-plane corpus candidate, written as IVF +
/// encoder-reconstruction YUV for external black-box decoder
/// validation and corpus pinning.
#[test]
fn scc_fixture_dump() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SCC_FIXTURE_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let input = vdelta_input();
    let enc = encode_key_frame_yuv420_with_q(&input, 60).expect("encode");
    std::fs::write(
        format!("{dir}/self-kf-64x64-q60-vdelta.ivf"),
        &enc.ivf_bytes,
    )
    .unwrap();
    let mut yuv = Vec::new();
    yuv.extend_from_slice(&enc.recon_y);
    yuv.extend_from_slice(&enc.recon_u);
    yuv.extend_from_slice(&enc.recon_v);
    std::fs::write(format!("{dir}/self-kf-64x64-q60-vdelta.yuv"), &yuv).unwrap();
    eprintln!("scc fixture dump: {} B IVF written", enc.ivf_bytes.len());
}
