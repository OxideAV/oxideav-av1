//! r482 — the in-band `Frame::StreamParams` event (oxideav-core
//! 0.1.43): the registry decoder returns one event immediately before
//! the first frame of every new output shape, after every frame of
//! the old one, under every worker budget.
//!
//! The `tests/fixtures/stream_params/*.ivf` streams come from
//! black-box `aomenc` / `SvtAv1EncApp` runs (`gen.sh` there, which
//! also prints the `ffprobe -show_frames` sizes and the reference
//! digests pinned below). Their decoded output is byte-identical to
//! `aomdec --rawvideo` and to `dav1d -o *.yuv`:
//!
//! * `aom_resize_random` — a random size per frame (§5.9.5
//!   `frame_size_override_flag`, §5.9.7 `frame_size_with_refs`,
//!   §7.11.3.3 scaled references);
//! * `aom_superres_random` — random §5.9.8 superres denominators only:
//!   the emitted size is the upscaled one (`UpscaledWidth ×
//!   FrameHeight`), which never changes, so no event;
//! * `aom_resize_superres` — superres on resized frames;
//! * `svt_resize_ra` — two scheduled resizes, each landing on a §7.21
//!   `show_existing_frame` output;
//! * `svt_resize_random` — a random size per frame under a
//!   hierarchical group (many `show_existing_frame` outputs at a new
//!   size).
//!
//! The emitted size is the decoded frame size (`UpscaledWidth ×
//! FrameHeight`), not the §5.9.6 render size: the render size is
//! presentation metadata, the planes are the frame size.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.5–§5.9.8, §7.16, §7.18.2,
//! §7.21.

use oxideav_av1::registry::{make_decoder, make_encoder};
use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, ExecutionContext, Frame, Packet, PixelFormat,
    TimeBase, VideoFrame, VideoPlane,
};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/stream_params");

/// One fixture: name, `ffprobe -show_frames` sizes in output order,
/// FNV-1a 64 of the reference decoder's output, and the output
/// indices that are §7.21 `show_existing_frame` units.
struct Case {
    name: &'static str,
    sizes: [(u32, u32); 16],
    digest: u64,
    show_existing: &'static [usize],
}

const CASES: [Case; 5] = [
    Case {
        name: "aom_resize_random",
        sizes: [
            (64, 43),
            (51, 34),
            (70, 47),
            (59, 39),
            (64, 43),
            (51, 34),
            (70, 47),
            (59, 39),
            (96, 64),
            (64, 43),
            (64, 43),
            (77, 51),
            (48, 32),
            (70, 47),
            (85, 57),
            (48, 32),
        ],
        digest: 0x10d6_9f09_a82b_af3d,
        show_existing: &[],
    },
    Case {
        name: "aom_superres_random",
        sizes: [(96, 64); 16],
        digest: 0x2223_b8c0_0327_b706,
        show_existing: &[],
    },
    Case {
        name: "aom_resize_superres",
        sizes: [
            (70, 47),
            (70, 47),
            (70, 47),
            (59, 39),
            (64, 43),
            (64, 43),
            (70, 47),
            (64, 43),
            (96, 64),
            (70, 47),
            (64, 43),
            (77, 51),
            (70, 47),
            (70, 47),
            (85, 57),
            (64, 43),
        ],
        digest: 0x44ac_b75a_6662_9f2f,
        show_existing: &[],
    },
    Case {
        name: "svt_resize_ra",
        sizes: [
            (96, 64),
            (96, 64),
            (96, 64),
            (96, 64),
            (64, 43),
            (64, 43),
            (64, 43),
            (64, 43),
            (64, 43),
            (64, 43),
            (48, 32),
            (48, 32),
            (48, 32),
            (48, 32),
            (48, 32),
            (48, 32),
        ],
        digest: 0x367e_8ddd_436d_d77c,
        show_existing: &[2, 4, 6, 8, 10, 12, 14],
    },
    Case {
        name: "svt_resize_random",
        sizes: [
            (64, 43),
            (64, 43),
            (59, 39),
            (51, 34),
            (70, 47),
            (59, 39),
            (70, 47),
            (96, 64),
            (51, 34),
            (77, 51),
            (64, 43),
            (48, 32),
            (64, 43),
            (85, 57),
            (70, 47),
            (48, 32),
        ],
        digest: 0x6002_25f8_a918_1b57,
        show_existing: &[2, 4, 6, 8, 10, 12, 14],
    },
];

fn temporal_units(name: &str) -> Vec<Vec<u8>> {
    let ivf = std::fs::read(format!("{FIX}/{name}.ivf")).expect("fixture readable");
    oxideav_av1::encoder::ivf::IvfReader::new(&ivf)
        .expect("ivf header")
        .read_all()
        .expect("ivf records")
        .into_iter()
        .map(|r| r.payload)
        .collect()
}

fn decoder(threads: usize) -> Box<dyn Decoder> {
    let mut d = make_decoder(&CodecParameters::video(CodecId::new("av1"))).expect("decoder");
    d.set_execution_context(&ExecutionContext::with_threads(threads));
    d
}

fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// What `receive_frame` returned, in order.
#[derive(Debug)]
enum Item {
    /// A `StreamParams` event: its payload's size and the generation
    /// read right after it.
    Event((u32, u32), u64),
    /// A video frame: its luma size (from the planes) and the
    /// generation read right after it.
    Video((u32, u32), u64, VideoFrame),
}

/// Drain everything ready, checking the per-call invariants: an
/// event's payload equals `output_params()`, the generation moved by
/// exactly one at the event and not at any frame, and a frame's planes
/// have exactly the size `output_params()` states.
fn drain(dec: &mut dyn Decoder, out: &mut Vec<Item>, ctx: &str) {
    loop {
        let before = dec.output_params_generation();
        match dec.receive_frame() {
            Ok(Frame::StreamParams(ev)) => {
                let now = dec.output_params().expect("described at an event");
                assert_eq!(format!("{ev:?}"), format!("{now:?}"), "{ctx}: payload");
                let g = dec.output_params_generation();
                assert_eq!(g, before.wrapping_add(1), "{ctx}: one bump per event");
                out.push(Item::Event((ev.width.unwrap(), ev.height.unwrap()), g));
            }
            Ok(Frame::Video(v)) => {
                let p = dec.output_params().expect("described after a frame");
                let g = dec.output_params_generation();
                assert_eq!(g, before, "{ctx}: a frame never moves the generation");
                let (w, h) = (p.width.unwrap(), p.height.unwrap());
                assert_eq!(p.pixel_format, Some(PixelFormat::Yuv420P), "{ctx}");
                assert_eq!(v.planes[0].stride, w as usize, "{ctx}: luma stride");
                assert_eq!(v.planes[0].data.len(), (w * h) as usize, "{ctx}: luma rows");
                let c = (w.div_ceil(2) * h.div_ceil(2)) as usize;
                assert_eq!(v.planes[1].data.len(), c, "{ctx}: chroma");
                out.push(Item::Video((w, h), g, v));
            }
            Ok(other) => panic!("{ctx}: unexpected {other:?}"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("{ctx}: receive_frame: {e}"),
        }
    }
}

/// Decode `units` one per packet, draining after each when
/// `streaming` (a player) or only after the flush (every frame in
/// flight at once).
fn run(units: &[Vec<u8>], threads: usize, streaming: bool, ctx: &str) -> Vec<Item> {
    let mut dec = decoder(threads);
    let mut out = Vec::new();
    for (i, u) in units.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), u.clone()).with_pts(i as i64))
            .expect("packet");
        if streaming {
            drain(dec.as_mut(), &mut out, ctx);
        }
    }
    dec.flush().unwrap();
    drain(dec.as_mut(), &mut out, ctx);
    assert_eq!(
        dec.receive_frame().unwrap_err().to_string(),
        Error::Eof.to_string()
    );
    out
}

/// Check one decode's item sequence against the fixture: the frames
/// are the reference output in order (sizes + digest + pts), and an
/// event precedes exactly the frames whose size differs from the
/// previous frame's, carrying that frame's size.
fn check(case: &Case, items: &[Item], ctx: &str) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut sizes = Vec::new();
    let mut pending: Option<(u32, u32)> = None;
    let mut events_before = Vec::new();
    let mut last_gen = 0;
    for item in items {
        match item {
            Item::Event(size, g) => {
                assert!(pending.is_none(), "{ctx}: two events in a row");
                assert!(!sizes.is_empty(), "{ctx}: an event for the first shape");
                assert_ne!(sizes.last(), Some(size), "{ctx}: an event without a change");
                pending = Some(*size);
                last_gen = *g;
            }
            Item::Video(size, g, v) => {
                let i = sizes.len();
                assert_eq!(v.pts, Some(i as i64), "{ctx}: output order");
                if let Some(announced) = pending.take() {
                    assert_eq!(announced, *size, "{ctx}: frame {i} has the announced size");
                    events_before.push(i);
                } else if let Some(prev) = sizes.last() {
                    assert_eq!(prev, size, "{ctx}: frame {i} changed size silently");
                }
                assert_eq!(*g, last_gen, "{ctx}: frame {i} generation");
                for p in &v.planes {
                    fnv1a(&mut h, &p.data);
                }
                sizes.push(*size);
            }
        }
    }
    assert!(pending.is_none(), "{ctx}: trailing event");
    assert_eq!(sizes, case.sizes, "{ctx}: per-frame sizes (ffprobe)");
    assert_eq!(h, case.digest, "{ctx}: output digest (reference decoder)");
    let expected: Vec<usize> = (1..16)
        .filter(|&i| case.sizes[i] != case.sizes[i - 1])
        .collect();
    assert_eq!(events_before, expected, "{ctx}: one event per size change");
    assert_eq!(
        last_gen,
        expected.len() as u64,
        "{ctx}: generation == events"
    );
}

#[test]
fn every_size_change_is_announced_once_for_every_worker_budget() {
    for case in &CASES {
        let units = temporal_units(case.name);
        assert_eq!(units.len(), 16, "{}: one shown frame per unit", case.name);
        for threads in [1usize, 2, 3, 4, 8] {
            for streaming in [true, false] {
                let ctx = format!("{} threads={threads} streaming={streaming}", case.name);
                check(case, &run(&units, threads, streaming, &ctx), &ctx);
            }
        }
    }
}

/// The superres-only stream codes frames at several downscaled widths
/// but emits every one at the upscaled size: a constant stream, no
/// event, generation 0.
#[test]
fn superres_alone_is_a_constant_shape() {
    let case = &CASES[1];
    let items = run(&temporal_units(case.name), 8, true, case.name);
    assert!(items.iter().all(|i| matches!(i, Item::Video(_, 0, _))));
    assert_eq!(items.len(), 16);
}

/// A §7.21 `show_existing_frame` output at a new size is announced
/// like any other frame: the SVT streams' size changes land on such
/// units (`svt_resize_ra`'s two resizes both do).
#[test]
fn show_existing_frame_outputs_of_a_new_size_are_announced() {
    let mut landed = 0;
    for case in &CASES[3..] {
        let items = run(&temporal_units(case.name), 4, true, case.name);
        let mut frame = 0;
        let mut announced = false;
        for item in &items {
            match item {
                Item::Event(..) => announced = true,
                Item::Video(..) => {
                    if announced && case.show_existing.contains(&frame) {
                        landed += 1;
                    }
                    announced = false;
                    frame += 1;
                }
            }
        }
    }
    // svt_resize_ra: frames 4 and 10; svt_resize_random: 2, 4, 6, 8,
    // 10, 12, 14 (each a new size).
    assert_eq!(landed, 2 + 7);
}

/// `reset()` (a seek) drops a frame held behind its event; the next
/// shape is compared against the announced one, so landing back on a
/// KEY frame of another size announces it, and landing on the same
/// size does not.
#[test]
fn reset_keeps_the_announced_shape_as_the_reference() {
    for threads in [1usize, 8] {
        let units = temporal_units("aom_resize_random");
        let mut dec = decoder(threads);
        // Frame 0 (64x43), then frame 1 (51x34) is announced.
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), units[0].clone()))
            .unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), units[1].clone()))
            .unwrap();
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
        let ev = dec.receive_frame().expect("event");
        assert_eq!(ev.stream_params().and_then(|p| p.width), Some(51));
        assert_eq!(dec.output_params_generation(), 1);
        // Seek before the held frame is taken: it is dropped.
        dec.reset().unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
        assert_eq!(
            dec.output_params().unwrap().width,
            Some(51),
            "announced shape kept"
        );
        // Land on the KEY frame again (64x43): a change from 51x34.
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), units[0].clone()))
            .unwrap();
        dec.flush().unwrap();
        let ev = dec.receive_frame().expect("event after the seek");
        assert_eq!(
            ev.stream_params().map(|p| (p.width, p.height)),
            Some((Some(64), Some(43)))
        );
        assert_eq!(dec.output_params_generation(), 2);
        let Ok(Frame::Video(v)) = dec.receive_frame() else {
            panic!("the announced frame");
        };
        assert_eq!(v.planes[0].stride, 64);
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
        // Same size again after another seek: no event.
        dec.reset().unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), units[0].clone()))
            .unwrap();
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
        assert_eq!(dec.output_params_generation(), 2);
    }
}

/// `receive_arena_frame`'s default skips the events; the counter still
/// moved once per change.
#[test]
fn arena_frames_skip_the_events() {
    let case = &CASES[3];
    let units = temporal_units(case.name);
    let mut dec = decoder(3);
    for u in &units {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), u.clone()))
            .unwrap();
    }
    dec.flush().unwrap();
    let mut n = 0;
    while let Ok(f) = dec.receive_arena_frame() {
        assert_eq!(f.header().width, case.sizes[n].0, "arena frame {n}");
        n += 1;
    }
    assert_eq!(n, 16);
    assert_eq!(dec.output_params_generation(), 2);
}

/// The encoder's coded size and sequence header are fixed at
/// construction: a mid-stream input-shape event is `Unsupported`, and
/// the encoder keeps working on frames of its configured shape.
#[test]
fn encoder_rejects_the_event_as_unsupported() {
    let mut p = CodecParameters::video(CodecId::new("av1"));
    p.width = Some(16);
    p.height = Some(16);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    let mut enc = make_encoder(&p).expect("encoder");
    let mut ev = CodecParameters::video(CodecId::new("av1"));
    ev.width = Some(32);
    ev.height = Some(32);
    ev.pixel_format = Some(PixelFormat::Yuv420P);
    let err = enc
        .send_frame(&Frame::StreamParams(Box::new(ev)))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    let plane = |w: usize| VideoPlane {
        stride: w,
        data: vec![128; w * w],
    };
    enc.send_frame(&Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![plane(16), plane(8), plane(8)],
    }))
    .expect("frame of the configured shape");
    enc.flush().unwrap();
    assert!(enc.receive_packet().is_ok());
}
