//! Runtime libavcodec interop for the AV1 cross-decode fuzz oracle.
//!
//! libavcodec is loaded via `dlopen` at first call — there is no
//! `ffmpeg-sys` / `ffmpeg-next` build-script dep that would pull
//! ffmpeg source into the workspace's cargo dep tree. The harness
//! checks `available()` up front and `eprintln!`s + early-returns when
//! the shared library isn't installed, so fuzz binaries built on a
//! host without ffmpeg simply do nothing instead of panicking.
//! There is **no** `#[ignore]` shortcut — runtime skip only.
//!
//! Workspace policy bars consulting ffmpeg / libavcodec / libdav1d /
//! libaom source; we only inspect the public C headers
//! (`<libavcodec/avcodec.h>`, `<libavcodec/packet.h>`,
//! `<libavutil/frame.h>`) for function signatures and the
//! `AV_CODEC_ID_AV1` enum value (= 226).
//!
//! Install on Debian / Ubuntu with `apt-get install -y ffmpeg`. The
//! loader probes the standard SONAMEs (`libavcodec.so.62`, `.so.61`,
//! `.so.60`, `.so`, `libavcodec.dylib`).

#![allow(unsafe_code)]

pub mod libavcodec {
    use libloading::{Library, Symbol};
    use std::ffi::{c_int, c_void};
    use std::sync::OnceLock;

    /// `AV_CODEC_ID_AV1` from `<libavcodec/codec_id.h>`. Stable since
    /// the AV1 enum entry was added in libavcodec 58 (2018).
    const AV_CODEC_ID_AV1: c_int = 226;

    /// libavcodec shared-library candidates the loader will try in
    /// order. Newest SONAMEs first so a host with multiple installs
    /// picks the modern one.
    const CANDIDATES: &[&str] = &[
        "libavcodec.so.62",
        "libavcodec.so.61",
        "libavcodec.so.60",
        "libavcodec.so.59",
        "libavcodec.so.58",
        "libavcodec.so",
        "libavcodec.dylib",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libavcodec is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            eprintln!("[oracle skip] libavcodec not available — install ffmpeg");
            None
        })
        .as_ref()
    }

    /// True iff a libavcodec shared library was successfully loaded.
    pub fn available() -> bool {
        lib().is_some()
    }

    /// An 8-bit 4:2:0 frame as decoded by libavcodec.
    pub struct DecodedFrame {
        pub width: u32,
        pub height: u32,
        /// libavutil pixel format id (`AVPixelFormat`). 0 == AV_PIX_FMT_YUV420P.
        pub pix_fmt: c_int,
        pub y: Vec<u8>,
        pub u: Vec<u8>,
        pub v: Vec<u8>,
        /// Plane width / height for chroma derived from pix_fmt
        /// subsampling. We only fill `u` / `v` when libavutil reports
        /// 4:2:0 (`AV_PIX_FMT_YUV420P` = 0).
        pub chroma_w: u32,
        pub chroma_h: u32,
    }

    /// Decode an AV1 OBU byte stream. Returns:
    ///
    /// * `Some(Ok(frames))` — libavcodec accepted the bitstream and
    ///   produced zero or more frames. Empty vec is a valid outcome
    ///   (e.g. show_existing_frame with no decoded refs).
    /// * `Some(Err(()))` — libavcodec returned an error (decoder
    ///   refused). The fuzz oracle treats this as "we don't have to
    ///   match libavcodec's behaviour here".
    /// * `None` — libavcodec is not available; the harness should skip.
    pub fn decode_av1(data: &[u8]) -> Option<core::result::Result<Vec<DecodedFrame>, ()>> {
        // Function pointer typedefs — signatures stable across libavcodec 58..62.
        // Documented in <libavcodec/avcodec.h>:
        //   const AVCodec *avcodec_find_decoder(enum AVCodecID id);
        //   AVCodecContext *avcodec_alloc_context3(const AVCodec *codec);
        //   int avcodec_open2(AVCodecContext *avctx, const AVCodec *codec, AVDictionary **options);
        //   AVPacket *av_packet_alloc(void);
        //   AVFrame *av_frame_alloc(void);
        //   int avcodec_send_packet(AVCodecContext *avctx, const AVPacket *avpkt);
        //   int avcodec_receive_frame(AVCodecContext *avctx, AVFrame *frame);
        //   void av_packet_free(AVPacket **pkt);
        //   void av_frame_free(AVFrame **frame);
        //   void avcodec_free_context(AVCodecContext **avctx);
        type FindDecoderFn = unsafe extern "C" fn(c_int) -> *const c_void;
        type AllocCtxFn = unsafe extern "C" fn(*const c_void) -> *mut c_void;
        type Open2Fn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) -> c_int;
        type PacketAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type FrameAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type SendPacketFn = unsafe extern "C" fn(*mut c_void, *const c_void) -> c_int;
        type RecvFrameFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
        type PacketFreeFn = unsafe extern "C" fn(*mut *mut c_void);
        type FrameFreeFn = unsafe extern "C" fn(*mut *mut c_void);
        type FreeCtxFn = unsafe extern "C" fn(*mut *mut c_void);

        let l = lib()?;
        unsafe {
            let find_decoder: Symbol<FindDecoderFn> = l.get(b"avcodec_find_decoder").ok()?;
            let alloc_ctx: Symbol<AllocCtxFn> = l.get(b"avcodec_alloc_context3").ok()?;
            let open2: Symbol<Open2Fn> = l.get(b"avcodec_open2").ok()?;
            let packet_alloc: Symbol<PacketAllocFn> = l.get(b"av_packet_alloc").ok()?;
            let frame_alloc: Symbol<FrameAllocFn> = l.get(b"av_frame_alloc").ok()?;
            let send_packet: Symbol<SendPacketFn> = l.get(b"avcodec_send_packet").ok()?;
            let recv_frame: Symbol<RecvFrameFn> = l.get(b"avcodec_receive_frame").ok()?;
            let packet_free: Symbol<PacketFreeFn> = l.get(b"av_packet_free").ok()?;
            let frame_free: Symbol<FrameFreeFn> = l.get(b"av_frame_free").ok()?;
            let free_ctx: Symbol<FreeCtxFn> = l.get(b"avcodec_free_context").ok()?;

            let codec = find_decoder(AV_CODEC_ID_AV1);
            if codec.is_null() {
                eprintln!("[oracle skip] libavcodec has no AV1 decoder linked");
                return None;
            }
            let mut ctx = alloc_ctx(codec);
            if ctx.is_null() {
                return Some(Err(()));
            }

            let mut frames: Vec<DecodedFrame> = Vec::new();
            let result = (|| -> core::result::Result<Vec<DecodedFrame>, ()> {
                if open2(ctx, codec, std::ptr::null_mut()) < 0 {
                    return Err(());
                }
                let mut pkt = packet_alloc();
                if pkt.is_null() {
                    return Err(());
                }
                let mut av_frame = frame_alloc();
                if av_frame.is_null() {
                    packet_free(&mut pkt);
                    return Err(());
                }

                // Populate AVPacket{ data, size, ... } via byte offsets.
                // AVPacket layout (libavcodec 58..62, see <libavcodec/packet.h>):
                //   off  0  AVBufferRef *buf
                //   off  8  i64 pts
                //   off 16  i64 dts
                //   off 24  u8 *data
                //   off 32  i32 size
                //   ...    (stream_index, flags, side_data, duration, pos, ...)
                // We only need to set `data` + `size`; libavcodec will
                // not free `data` because `buf == NULL` (ref-count
                // bypass for caller-owned buffers).
                let pkt_bytes = pkt as *mut u8;
                let pkt_data_ptr = pkt_bytes.add(24) as *mut *const u8;
                let pkt_size_ptr = pkt_bytes.add(32) as *mut c_int;
                pkt_data_ptr.write_unaligned(data.as_ptr());
                pkt_size_ptr.write_unaligned(data.len() as c_int);

                let send_rc = send_packet(ctx, pkt);
                if send_rc < 0 {
                    av_frame_free_safe(&frame_free, &mut av_frame);
                    packet_free(&mut pkt);
                    return Err(());
                }

                // Drain any frames that came out of this packet.
                for _ in 0..256 {
                    let rc = recv_frame(ctx, av_frame);
                    if rc < 0 {
                        // -EAGAIN / -EOF — no more frames now.
                        break;
                    }
                    if let Some(df) = read_avframe(av_frame) {
                        frames.push(df);
                    }
                }

                // Send a NULL packet (drain mode).
                pkt_data_ptr.write_unaligned(std::ptr::null());
                pkt_size_ptr.write_unaligned(0);
                let _ = send_packet(ctx, pkt);
                for _ in 0..256 {
                    let rc = recv_frame(ctx, av_frame);
                    if rc < 0 {
                        break;
                    }
                    if let Some(df) = read_avframe(av_frame) {
                        frames.push(df);
                    }
                }

                av_frame_free_safe(&frame_free, &mut av_frame);
                packet_free(&mut pkt);
                Ok(frames)
            })();
            free_ctx(&mut ctx);
            Some(result)
        }
    }

    /// Convenience: free an AVFrame guarded against double-free.
    unsafe fn av_frame_free_safe(
        free_fn: &Symbol<unsafe extern "C" fn(*mut *mut c_void)>,
        frame: &mut *mut c_void,
    ) {
        if !frame.is_null() {
            free_fn(frame);
        }
    }

    /// Read an `AVFrame` into our owned `DecodedFrame` representation.
    /// Returns None if the frame has unsupported pix_fmt or zero
    /// dimensions.
    ///
    /// AVFrame layout (libavcodec 58..62, see `<libavutil/frame.h>`):
    ///   off    0  uint8_t *data[8]
    ///   off   64  int linesize[8]
    ///   off   96  uint8_t **extended_data
    ///   off  104  int width
    ///   off  108  int height
    ///   off  112  int nb_samples
    ///   off  116  int format
    /// We read these by byte offset (matching the layout shipped in
    /// libavutil 56-58; same offsets as long as the leading fields
    /// stay in their documented order).
    unsafe fn read_avframe(frame_ptr: *mut c_void) -> Option<DecodedFrame> {
        if frame_ptr.is_null() {
            return None;
        }
        let bytes = frame_ptr as *const u8;
        let data_ptr = bytes as *const *const u8;
        let linesize_ptr = bytes.add(64) as *const c_int;
        let width = (bytes.add(104) as *const c_int).read_unaligned();
        let height = (bytes.add(108) as *const c_int).read_unaligned();
        let format = (bytes.add(116) as *const c_int).read_unaligned();

        if width <= 0 || height <= 0 {
            return None;
        }
        // AV_PIX_FMT_YUV420P == 0; AV_PIX_FMT_YUV422P == 4;
        // AV_PIX_FMT_YUV444P == 5; AV_PIX_FMT_GRAY8 == 8.
        // We copy luma always; chroma only for the 4:2:0 fast path
        // (the most common AV1 output).
        let y_data = data_ptr.read_unaligned();
        let y_stride = linesize_ptr.read_unaligned();
        if y_data.is_null() || y_stride <= 0 {
            return None;
        }
        let w = width as usize;
        let h = height as usize;
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            let src = y_data.add(row * y_stride as usize);
            std::ptr::copy_nonoverlapping(src, y.as_mut_ptr().add(row * w), w);
        }

        // Chroma: only fill for AV_PIX_FMT_YUV420P (format == 0).
        let (chroma_w, chroma_h, mut u, mut v) = if format == 0 {
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            (cw as u32, ch as u32, vec![0u8; cw * ch], vec![0u8; cw * ch])
        } else {
            (0u32, 0u32, Vec::new(), Vec::new())
        };
        if format == 0 {
            let u_data = data_ptr.add(1).read_unaligned();
            let v_data = data_ptr.add(2).read_unaligned();
            let u_stride = linesize_ptr.add(1).read_unaligned();
            let v_stride = linesize_ptr.add(2).read_unaligned();
            if !u_data.is_null() && !v_data.is_null() && u_stride > 0 && v_stride > 0 {
                for row in 0..(chroma_h as usize) {
                    let su = u_data.add(row * u_stride as usize);
                    let sv = v_data.add(row * v_stride as usize);
                    std::ptr::copy_nonoverlapping(
                        su,
                        u.as_mut_ptr().add(row * chroma_w as usize),
                        chroma_w as usize,
                    );
                    std::ptr::copy_nonoverlapping(
                        sv,
                        v.as_mut_ptr().add(row * chroma_w as usize),
                        chroma_w as usize,
                    );
                }
            } else {
                u.clear();
                v.clear();
            }
        }

        Some(DecodedFrame {
            width: width as u32,
            height: height as u32,
            pix_fmt: format,
            y,
            u,
            v,
            chroma_w,
            chroma_h,
        })
    }
}
