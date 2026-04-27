//! r19 chain-walk diagnostic for SVT-AV1 fixtures.
//!
//! Walks every Frame OBU in `/tmp/av1_inter.ivf` (a 48-frame SVT-AV1
//! fixture; see `tests/inter_decode.rs` for the regen recipe) with
//! the §7.20 DPB refresh chain wired through (`refresh_with_gm`) and
//! reports `(parsed_ok, total)`. The fixture, when present, is
//! required to surface at least the round-18 baseline (38/48 Frame
//! OBUs); subsequent rounds tighten the bound as more spec edges are
//! covered.
//!
//! When the fixture is missing the test is a no-op so CI doesn't need
//! ffmpeg / libsvtav1 installed.
//!
//! Why this test exists: the README r18 line cited "SVT-AV1 35/44
//! Frame OBUs (chain prerequisite for r19)" with no in-tree assertion
//! capturing the metric. r19 adds this so any regression in the
//! parsing chain (frame_header bit accounting, DPB save/load of
//! `gm_params`, `set_frame_refs()` understanding) shows up
//! immediately, even before the deeper "decode every frame to pixels"
//! tests can be brought back online.

use std::path::Path;

use oxideav_av1::dpb::Dpb;
use oxideav_av1::frame_header::{parse_frame_obu_with_dpb, FrameType};
use oxideav_av1::{iter_obus, parse_sequence_header, ObuType};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("svtav1_chain_walk: fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

fn ivf_packet_slices(data: &[u8]) -> Option<Vec<Vec<u8>>> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return None;
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut pos = header_len;
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12;
        if pos + size > data.len() {
            break;
        }
        out.push(data[pos..pos + size].to_vec());
        pos += size;
    }
    Some(out)
}

/// Walk every Frame OBU across the IVF, refreshing the DPB with the
/// just-decoded `gm_params` per §7.20 so primary_ref_frame chains see
/// the right `PrevGmParams[]` values. Asserts that we parse at least
/// 38 frames OK out of 48 — the round-18 baseline. Failures of
/// "out of bits" inside `parse_global_motion_params` tend to
/// indicate either a missing `set_frame_refs()` (§7.8) implementation
/// or a downstream bit-account miscount; both are tracked as r19+
/// pending items.
#[test]
fn svtav1_chain_walk_baseline_38_of_48() {
    let Some(data) = read_fixture("/tmp/av1_inter.ivf") else {
        return;
    };
    let pkts = ivf_packet_slices(&data).expect("ivf parse");

    let mut seq = None;
    for o in iter_obus(&pkts[0]) {
        let o = o.expect("iter pkt0");
        if o.header.obu_type == ObuType::SequenceHeader {
            seq = parse_sequence_header(o.payload).ok();
        }
    }
    let seq = seq.expect("seq header in pkt 0");

    let mut dpb = Dpb::new();
    let mut total = 0u32;
    let mut ok = 0u32;
    let mut first_fail: Option<(usize, u32)> = None;

    for (pi, pkt) in pkts.iter().enumerate() {
        for o in iter_obus(pkt) {
            let Ok(o) = o else { continue };
            if o.header.obu_type != ObuType::Frame {
                continue;
            }
            total += 1;
            match parse_frame_obu_with_dpb(&seq, o.payload, &dpb) {
                Ok((fh, _)) => {
                    ok += 1;
                    if fh.frame_type == FrameType::Key {
                        dpb.reset();
                    }
                    if fh.refresh_frame_flags != 0 {
                        // §7.20: refresh OrderHint + saved gm_params
                        // together so primary_ref_frame chains can
                        // load the just-decoded warp matrix.
                        dpb.refresh_with_gm(
                            fh.refresh_frame_flags,
                            fh.order_hint,
                            None,
                            &fh.gm_params,
                        );
                    }
                }
                Err(_) => {
                    if first_fail.is_none() {
                        first_fail = Some((pi, total));
                    }
                }
            }
        }
    }

    eprintln!("svtav1_chain_walk: parsed {ok}/{total} Frame OBUs");
    if let Some((pi, n)) = first_fail {
        eprintln!("svtav1_chain_walk: first fail = pkt {pi} Frame #{n}");
    }
    assert!(
        ok >= 38,
        "regressed below round-18 baseline: only {ok}/{total} Frame OBUs parsed"
    );
    // Total varies with the fixture — accept any plausible length so
    // CI doesn't break on encoder quirks across libsvtav1 versions.
    assert!(
        total >= 44,
        "fixture has too few Frame OBUs ({total}); regenerate /tmp/av1_inter.ivf with at least 2s of input"
    );
}
