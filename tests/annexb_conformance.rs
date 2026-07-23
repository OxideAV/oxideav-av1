//! r428 — Annex B length-delimited bitstream conformance
//! (`crate::annexb` + `decoder::decode_av1_annexb`).
//!
//! Three gates:
//!
//! 1. **External pin** — an independently produced Annex B stream
//!    (black-box external encoder, `--annexb` output) must decode
//!    through [`oxideav_av1::decoder::decode_av1_annexb`] to the
//!    SHA-256 digest THREE independent black-box reference decoders
//!    agree on (each fed the Annex B stream directly).
//! 2. **Repack equivalence** — this crate's own conformance-grade
//!    streams (KEY / inter GOP / pyramid), rewrapped from their §7.5
//!    low-overhead temporal units into Annex B framing by
//!    [`oxideav_av1::annexb::build_from_temporal_units`], must decode
//!    through the Annex B entry to the SAME pixels as the IVF path —
//!    and byte-exactly to the encoder reconstruction.
//! 3. **Framing hygiene** — the Annex B.2/B.3 validation surface
//!    (unit tests in `src/annexb.rs` cover the structural rules;
//!    the malformed-input test here covers the public entry).

use oxideav_av1::annexb::build_from_temporal_units;
use oxideav_av1::decoder::{decode_av1_annexb, Frame};
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q, encode_key_frame_yuv420_with_q, encode_pyramid_gop_yuv420_with_q,
    Yuv420Frame,
};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    // Minimal FIPS 180-4 SHA-256 (same self-checked shape the
    // fixture-conformance suite carries).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = bytes.to_vec();
    let bitlen = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

/// SHA-256 self-check against the FIPS 180-4 "abc" vector.
#[test]
fn sha256_matches_fips_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

// ---------------------------------------------------------------------
// Gate 1: the external Annex B pin.
// ---------------------------------------------------------------------

/// An independently produced 96×80 4-frame Annex B stream (black-box
/// external encoder, `--annexb` output; notes under
/// `docs/video/av1/fixtures/ext-annexb-96x80/`).
const EXT_ANNEXB_96X80_OBU: &str = concat!(
    "a802a60201100b080000000335f9e6d7cc0296023010008f60c596208125960000b6625f9ef6",
    "7639c332230545a16d2211fc84425a19b1438181582ef7711c80ae8621809537b666ea66dcc9",
    "c66399ddc599aa8eba49fb79374c73a6cde40dbf2bb3c939f6b1c125b875092b49866350637d",
    "88c517648679db1b06c23f65b78cc6ace700253b9ea0a08bcb85479d40551209131d661fa161",
    "bdffbee1cbf0268b762ed4525d4abdbfccd58c493f3f637ba6561757d7444dadd842c9ee44eb",
    "5e37f575fe94d8120421a766d6e015474d795cfa06a6709bedf00d02c3ba99469d639cdee13d",
    "54ea3ab7af9ee50288ffc606dec1059370bd54a1f5c34bfc1c02dd20b11a2e5b34ceff8f7abc",
    "1f41aceaba378ccceb45eef87943792bbb06a81433da6cc150db38433445531ad901ae010110",
    "aa01302003e040000003493062cb2080a2cb802000d892311c666fb35e60aa450f511d45bbc5",
    "b63f26d8422441616791ec3477505bc12f49d5d463e2543831155d3e8a483bc07532e6250d42",
    "4400290a10d7fbc52f0e3624dd457d6b0d363d41e607069c8435a495efeefa91fb31a085ba9b",
    "7ddf735b1669c4cba350b0cb09b98803520632b072e7b456fdcd9c0ef21f058f952ada5b673e",
    "f8c479e33348fc473f2b56cc4f56ac14dcd590d02827303003c1000040069280228a080a00b0",
    "05007c8e365bab3e1a1455a577247e395797869e8fa620191801101530300402080001069400",
    "1a69860a000007009ffb401918011015303006000d00010694001249860a00000200a28315",
);

/// The Annex B entry must reproduce the digest THREE independent
/// black-box reference decoders produce from the same stream.
#[test]
fn external_annexb_stream_decodes_byte_exact() {
    let obu = unhex(EXT_ANNEXB_96X80_OBU);
    let frames = decode_av1_annexb(&obu).expect("annexb decode");
    assert_eq!(frames.len(), 4, "shown frame count");
    let mut out: Vec<u8> = Vec::new();
    for f in &frames {
        let Frame::Spec(s) = f else {
            panic!("non-Spec frame variant");
        };
        for p in &s.planes {
            out.extend_from_slice(p);
        }
    }
    assert_eq!(
        sha256_hex(&out),
        "df8d5cc55e9dd9e098006d3b22b3dbdda70c6f3fa038cc5f6b47d685c81b0905",
        "decoded pixels differ from the triple-validated reference digest"
    );
}

// ---------------------------------------------------------------------
// Gate 2: repack equivalence on this crate's own streams.
// ---------------------------------------------------------------------

fn moving(w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    (0..n)
        .map(|k| {
            let (wu, hu) = (w as usize, h as usize);
            let mut f = Yuv420Frame::filled(w, h, 0);
            for r in 0..hu {
                for c in 0..wu {
                    f.y[r * wu + c] = ((r * 5 + c * 3 + k * 7 + (r / 16) * (c / 16)) % 256) as u8;
                }
            }
            let (cw, ch) = (wu / 2, hu / 2);
            for r in 0..ch {
                for c in 0..cw {
                    f.u[r * cw + c] = ((128 + r * 2 + c + k) % 256) as u8;
                    f.v[r * cw + c] = ((64 + r + c * 2 + 2 * k) % 256) as u8;
                }
            }
            f
        })
        .collect()
}

fn assert_annexb_matches_recon(
    temporal_units: &[Vec<u8>],
    recon: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    label: &str,
) -> Vec<u8> {
    let annexb = build_from_temporal_units(temporal_units)
        .unwrap_or_else(|e| panic!("{label}: annexb build failed: {e:?}"));
    let frames = decode_av1_annexb(&annexb)
        .unwrap_or_else(|e| panic!("{label}: annexb decode failed: {e:?}"));
    assert_eq!(frames.len(), recon.len(), "{label}: shown frame count");
    for (idx, f) in frames.iter().enumerate() {
        let Frame::Spec(s) = f else {
            panic!("{label}: non-Spec frame");
        };
        assert_eq!(s.planes[0], recon[idx].0, "{label} frame {idx} luma");
        assert_eq!(s.planes[1], recon[idx].1, "{label} frame {idx} U");
        assert_eq!(s.planes[2], recon[idx].2, "{label} frame {idx} V");
    }
    annexb
}

/// KEY-only stream repacked to Annex B decodes to the encoder
/// reconstruction.
#[test]
fn key_repack_round_trips() {
    let f = &moving(96, 80, 1)[0];
    for q in [0u8, 100] {
        let enc = encode_key_frame_yuv420_with_q(f, q).expect("key encode");
        assert_annexb_matches_recon(
            std::slice::from_ref(&enc.temporal_unit_bytes),
            &[(
                enc.recon_y.clone(),
                enc.recon_u.clone(),
                enc.recon_v.clone(),
            )],
            "key",
        );
    }
}

/// Inter GOP repacked to Annex B: one frame unit per frame, decodes
/// to the per-frame reconstructions. Dumps the witness stream when
/// `OXIDEAV_AV1_ANNEXB_DIR` is set (the docs-staged fixture).
#[test]
fn gop_repack_round_trips() {
    let frames = moving(96, 80, 4);
    let enc = encode_gop_yuv420_with_q(&frames, 80).expect("gop encode");
    let recon: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = enc
        .recon
        .iter()
        .map(|r| (r.y.clone(), r.u.clone(), r.v.clone()))
        .collect();
    let annexb = assert_annexb_matches_recon(&enc.temporal_units, &recon, "gop");
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_ANNEXB_DIR") {
        let root = std::path::Path::new(&dir);
        std::fs::create_dir_all(root).expect("create out dir");
        std::fs::write(root.join("self-annexb-96x80-q80.obu"), &annexb).expect("write obu");
        let mut yuv: Vec<u8> = Vec::new();
        for (y, u, v) in &recon {
            yuv.extend_from_slice(y);
            yuv.extend_from_slice(u);
            yuv.extend_from_slice(v);
        }
        std::fs::write(root.join("self-annexb-96x80-q80.yuv"), &yuv).expect("write yuv");
    }
}

/// Pyramid GOP repacked to Annex B: a temporal unit carrying
/// decoded-not-shown frames splits into MULTIPLE frame units (one
/// per frame, per Annex B.3) and still decodes in display order to
/// the reconstructions.
#[test]
fn pyramid_repack_round_trips() {
    let frames = moving(64, 64, 5);
    let enc = encode_pyramid_gop_yuv420_with_q(&frames, 80).expect("pyramid encode");
    let recon: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = enc
        .recon
        .iter()
        .map(|r| (r.y.clone(), r.u.clone(), r.v.clone()))
        .collect();
    assert_annexb_matches_recon(&enc.temporal_units, &recon, "pyramid");
}

// ---------------------------------------------------------------------
// Gate 3: public-entry framing hygiene.
// ---------------------------------------------------------------------

/// Malformed Annex B inputs surface typed errors, never panics.
#[test]
fn malformed_annexb_errors_cleanly() {
    assert!(
        decode_av1_annexb(&[]).unwrap().is_empty(),
        "empty = zero units"
    );
    assert!(decode_av1_annexb(&[0x05, 1, 2]).is_err(), "truncated TU");
    let obu = unhex(EXT_ANNEXB_96X80_OBU);
    for cut in 1..obu.len().min(64) {
        assert!(decode_av1_annexb(&obu[..cut]).is_err(), "cut {cut}");
    }
    // An IVF stream fed to the Annex B entry must be rejected (the
    // "DKIF" magic parses as nonsense lengths).
    let f = moving(16, 16, 1).remove(0);
    let ivf = encode_key_frame_yuv420_with_q(&f, 0).unwrap().ivf_bytes;
    assert!(decode_av1_annexb(&ivf).is_err(), "IVF is not Annex B");
}
