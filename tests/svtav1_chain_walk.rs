//! r19 chain-walk diagnostic for SVT-AV1 fixtures (post-r21 baseline).
//!
//! Walks every Frame OBU in `/tmp/av1_inter.ivf` (a 48-frame SVT-AV1
//! fixture; see `tests/inter_decode.rs` for the regen recipe) with
//! the §7.20 DPB refresh chain wired through (`refresh_with_gm`) and
//! reports `(parsed_ok, total)`. The fixture, when present, must
//! parse 100% (48/48) — round 21 fixed the §5.9.2 inter-branch order
//! (frame_size + render_size now run AFTER the ref_frame_idx loop) so
//! the parser no longer mis-aligns the bitstream by ~13 bits on every
//! non-short-signaling inter frame.
//!
//! When the fixture is missing the test is a no-op so CI doesn't need
//! ffmpeg / libsvtav1 installed.
//!
//! `AV1_TRACE_BITS=1` enables a per-OBU diagnostic trail (sequence
//! header config, packet OBU layout, per-frame bit checkpoints) that
//! made the r21 bisect possible; gated off in normal runs to keep the
//! output clean.

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
/// the right `PrevGmParams[]` values. Asserts that we parse 100%
/// (48/48) — the round-21 baseline. The §5.9.2 inter-branch order
/// fix unblocked the 10 frames that previously over-read past the
/// OBU end inside `parse_global_motion_params` / `parse_lr_params`.
/// Any regression of frame_size/render_size placement (or §5.9.7
/// frame_size_with_refs found_ref bit count) trips immediately.
#[test]
fn svtav1_chain_walk_round21_full_pass() {
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
    if std::env::var("AV1_TRACE_BITS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
    {
        eprintln!(
            "AV1_TRACE_BITS: seq enable_cdef={} enable_restoration={} use_128={} sx={} sy={} planes={} mono={} order_hint_bits={} enable_order_hint={} enable_warped_motion={} sep_uv={} enable_superres={} enable_ref_frame_mvs={} reduced_still={} fid_present={} fwbits={} fhbits={} max_w={} max_h={}",
            seq.enable_cdef,
            seq.enable_restoration,
            seq.use_128x128_superblock,
            seq.color_config.subsampling_x,
            seq.color_config.subsampling_y,
            seq.color_config.num_planes,
            seq.color_config.mono_chrome,
            seq.order_hint_bits,
            seq.enable_order_hint,
            seq.enable_warped_motion,
            seq.color_config.separate_uv_deltas,
            seq.enable_superres,
            seq.enable_ref_frame_mvs,
            seq.reduced_still_picture_header,
            seq.frame_id_numbers_present,
            seq.frame_width_bits,
            seq.frame_height_bits,
            seq.max_frame_width,
            seq.max_frame_height,
        );
        eprintln!(
            "AV1_TRACE_BITS: seq force_screen={} force_intmv={} fgp_present={}",
            seq.seq_force_screen_content_tools,
            seq.seq_force_integer_mv,
            seq.film_grain_params_present,
        );
    }

    let mut dpb = Dpb::new();
    let mut total = 0u32;
    let mut ok = 0u32;
    let mut first_fail: Option<(usize, u32)> = None;

    for (pi, pkt) in pkts.iter().enumerate() {
        if std::env::var("AV1_TRACE_BITS")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
        {
            let mut types = Vec::new();
            for o in iter_obus(pkt).flatten() {
                types.push(format!("{:?}({}b)", o.header.obu_type, o.payload.len()));
            }
            eprintln!(
                "AV1_TRACE_BITS: pkt#{} ({} bytes) OBUs: {}",
                pi,
                pkt.len(),
                types.join(",")
            );
        }
        for o in iter_obus(pkt) {
            let Ok(o) = o else { continue };
            if o.header.obu_type != ObuType::Frame {
                continue;
            }
            total += 1;
            if std::env::var("AV1_TRACE_BITS")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false)
            {
                let mut bytes = String::new();
                for b in o.payload {
                    bytes.push_str(&format!("{:02x}", b));
                }
                eprintln!(
                    "AV1_TRACE_BITS: frame#{} payload={} bytes [{}]",
                    total,
                    o.payload.len(),
                    bytes,
                );
            }
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
    // Round 21 (post-fix): the §5.9.2 inter-branch order fix unblocks all
    // 48 Frame OBUs in the SVT-AV1 fixture. Pin the new floor so any
    // regression of the frame_size/render_size placement (or the §5.9.7
    // frame_size_with_refs found_ref bit count) trips immediately.
    assert!(
        ok >= 48,
        "regressed below round-21 baseline: only {ok}/{total} Frame OBUs parsed"
    );
    // Total varies with the fixture — accept any plausible length so
    // CI doesn't break on encoder quirks across libsvtav1 versions.
    assert!(
        total >= 44,
        "fixture has too few Frame OBUs ({total}); regenerate /tmp/av1_inter.ivf with at least 2s of input"
    );
}
