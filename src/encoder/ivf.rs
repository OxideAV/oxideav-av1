//! IVF v0 container writer.
//!
//! IVF is a trivial public file format developed for VP8 testing
//! and reused for almost every On2 / WebM / AV1 reference fixture.
//! Its byte layout is so small the entire spec fits inline:
//!
//! ```text
//! File header — 32 bytes, little-endian:
//!     0..4   "DKIF"                 magic
//!     4..6   uint16  version        always 0
//!     6..8   uint16  header_length  always 32
//!     8..12  uint32  codec FOURCC   e.g. b"AV01"
//!    12..14  uint16  width          pixels
//!    14..16  uint16  height         pixels
//!    16..20  uint32  fps_num        timebase numerator
//!    20..24  uint32  fps_den        timebase denominator
//!    24..28  uint32  frame_count    may be left at 0 + patched
//!    28..32  uint32  unused / 0
//!
//! Per-frame header — 12 bytes, little-endian:
//!     0..4   uint32  frame_size     payload bytes (excludes header)
//!     4..12  uint64  pts            64-bit presentation timestamp
//!
//! Frame payload immediately follows the 12-byte frame header.
//! ```
//!
//! The layout is shared by every `.ivf` fixture already in
//! `docs/video/av1/fixtures/`; the file header of
//! `tiny-i-only-16x16-prof0/input.ivf` decodes byte-for-byte to
//! the table above with `version=0`, `header_length=32`,
//! `codec=AV01`, `16x16` dimensions, `fps_num=25`, `fps_den=1`,
//! `frame_count=1`. No external library source consulted — the
//! byte layout itself is the spec.
//!
//! The writer wraps any `std::io::Write` sink. Frame count is
//! tracked in-memory as `frames_written()`; callers that need the
//! count baked into the file header have two options:
//!
//!   1. Wrap a `std::io::Cursor<Vec<u8>>`, write all frames, then
//!      patch bytes 24..28 of the resulting buffer.
//!   2. Pass the final count to [`IvfWriter::patch_frame_count`]
//!      when the sink also implements `Seek`.

use std::io::{Result as IoResult, Seek, SeekFrom, Write};

/// `b"AV01"` — the IVF codec FOURCC for AV1, matching the file
/// header of every `.ivf` fixture under `docs/video/av1/fixtures/`.
pub const FOURCC_AV01: [u8; 4] = *b"AV01";

/// `b"VP80"` — common alternate. Included so the writer is not
/// AV1-specific in case it lands a reuse in a sibling crate later.
pub const FOURCC_VP80: [u8; 4] = *b"VP80";

/// `b"VP90"` — common alternate.
pub const FOURCC_VP90: [u8; 4] = *b"VP90";

/// Length of the IVF v0 file header (`header_length` field value).
pub const IVF_FILE_HEADER_LEN: usize = 32;

/// Length of each per-frame IVF header.
pub const IVF_FRAME_HEADER_LEN: usize = 12;

/// IVF v0 writer.
///
/// Construct with [`IvfWriter::new`]; the file header is emitted
/// immediately. Each [`IvfWriter::write_frame`] call appends a
/// 12-byte frame header followed by the payload bytes. The
/// in-memory frame count is updated after each successful write;
/// callers that want the count baked into the file header can
/// either patch it post-hoc (see [`IvfWriter::patch_frame_count`])
/// or seek the sink themselves.
#[derive(Debug)]
pub struct IvfWriter<W: Write> {
    sink: W,
    fourcc: [u8; 4],
    width: u16,
    height: u16,
    fps_num: u32,
    fps_den: u32,
    frames_written: u32,
}

impl<W: Write> IvfWriter<W> {
    /// Create a writer and emit the 32-byte file header.
    ///
    /// `fps_num` / `fps_den` are the IVF timebase rational; `pts`
    /// values supplied to [`Self::write_frame`] are in units of
    /// `fps_den / fps_num` seconds (the inverse of the rational —
    /// IVF stores pts as ticks of `1 / time_base`).
    pub fn new(
        mut sink: W,
        fourcc: [u8; 4],
        width: u16,
        height: u16,
        fps_num: u32,
        fps_den: u32,
    ) -> IoResult<Self> {
        let header = build_file_header(fourcc, width, height, fps_num, fps_den, 0);
        sink.write_all(&header)?;
        Ok(Self {
            sink,
            fourcc,
            width,
            height,
            fps_num,
            fps_den,
            frames_written: 0,
        })
    }

    /// Append one frame: 12-byte header (`u32 size` + `u64 pts`)
    /// followed by the payload bytes.
    pub fn write_frame(&mut self, payload: &[u8], pts: u64) -> IoResult<()> {
        let header = build_frame_header(payload.len() as u32, pts);
        self.sink.write_all(&header)?;
        self.sink.write_all(payload)?;
        self.frames_written = self.frames_written.saturating_add(1);
        Ok(())
    }

    /// Total number of frames written so far (in-memory counter).
    pub fn frames_written(&self) -> u32 {
        self.frames_written
    }

    /// Borrowed view of the FOURCC the file header was emitted with.
    pub fn fourcc(&self) -> [u8; 4] {
        self.fourcc
    }

    /// Dimensions baked into the file header.
    pub fn dimensions(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Timebase rational baked into the file header.
    pub fn timebase(&self) -> (u32, u32) {
        (self.fps_num, self.fps_den)
    }

    /// Consume the writer and return the underlying sink.
    pub fn into_inner(self) -> W {
        self.sink
    }
}

impl<W: Write + Seek> IvfWriter<W> {
    /// Patch the file header's `frame_count` field (bytes 24..28)
    /// with [`Self::frames_written`] and return the sink to the
    /// end of the file. Only available when the sink is seekable.
    pub fn patch_frame_count(&mut self) -> IoResult<()> {
        let count = self.frames_written;
        self.sink.seek(SeekFrom::Start(24))?;
        self.sink.write_all(&count.to_le_bytes())?;
        self.sink.seek(SeekFrom::End(0))?;
        Ok(())
    }
}

/// Build the 32-byte IVF v0 file header as a fixed buffer. Public
/// for callers that prefer to manage the sink themselves.
pub fn build_file_header(
    fourcc: [u8; 4],
    width: u16,
    height: u16,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
) -> [u8; IVF_FILE_HEADER_LEN] {
    let mut h = [0u8; IVF_FILE_HEADER_LEN];
    h[0..4].copy_from_slice(b"DKIF");
    h[4..6].copy_from_slice(&0u16.to_le_bytes()); // version
    h[6..8].copy_from_slice(&(IVF_FILE_HEADER_LEN as u16).to_le_bytes()); // header_length
    h[8..12].copy_from_slice(&fourcc);
    h[12..14].copy_from_slice(&width.to_le_bytes());
    h[14..16].copy_from_slice(&height.to_le_bytes());
    h[16..20].copy_from_slice(&fps_num.to_le_bytes());
    h[20..24].copy_from_slice(&fps_den.to_le_bytes());
    h[24..28].copy_from_slice(&frame_count.to_le_bytes());
    // bytes 28..32 stay 0 (unused).
    h
}

/// Build the 12-byte per-frame IVF header.
pub fn build_frame_header(frame_size: u32, pts: u64) -> [u8; IVF_FRAME_HEADER_LEN] {
    let mut h = [0u8; IVF_FRAME_HEADER_LEN];
    h[0..4].copy_from_slice(&frame_size.to_le_bytes());
    h[4..12].copy_from_slice(&pts.to_le_bytes());
    h
}

/// One frame demuxed from an IVF v0 file. Owns its payload bytes so
/// callers can drop the source buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfFrame {
    /// 64-bit presentation timestamp in `1 / time_base` ticks per the
    /// IVF v0 layout — see the module docs.
    pub pts: u64,
    /// Frame payload bytes — the encoder's §7.5 temporal-unit body
    /// (TD + optional SH + per-frame OBU sequence) in the AV1 case.
    pub payload: Vec<u8>,
}

/// Parsed IVF v0 file header. Construct via [`parse_file_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfFileHeader {
    /// `version` field at bytes 4..6. The v0 layout has `version == 0`.
    pub version: u16,
    /// `header_length` field at bytes 6..8. The v0 layout has
    /// `header_length == 32`.
    pub header_length: u16,
    /// `codec` FOURCC at bytes 8..12 (e.g. `b"AV01"`).
    pub fourcc: [u8; 4],
    /// `width` (pixels) at bytes 12..14.
    pub width: u16,
    /// `height` (pixels) at bytes 14..16.
    pub height: u16,
    /// `fps_num` (timebase numerator) at bytes 16..20.
    pub fps_num: u32,
    /// `fps_den` (timebase denominator) at bytes 20..24.
    pub fps_den: u32,
    /// `frame_count` at bytes 24..28. Encoders that don't track the
    /// final count up front leave this at `0`.
    pub frame_count: u32,
}

/// IVF v0 demuxer error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvfReadError {
    /// Buffer ended mid-header or mid-payload.
    UnexpectedEnd,
    /// File header magic was not `b"DKIF"`.
    BadMagic,
    /// File header `header_length` was not 32 (only the v0 layout is
    /// supported).
    UnsupportedHeaderLength(u16),
    /// File header `version` was not `0`.
    UnsupportedVersion(u16),
}

/// Parse the 32-byte IVF v0 file header. Returns the populated
/// [`IvfFileHeader`] on success.
pub fn parse_file_header(bytes: &[u8]) -> Result<IvfFileHeader, IvfReadError> {
    if bytes.len() < IVF_FILE_HEADER_LEN {
        return Err(IvfReadError::UnexpectedEnd);
    }
    if &bytes[0..4] != b"DKIF" {
        return Err(IvfReadError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != 0 {
        return Err(IvfReadError::UnsupportedVersion(version));
    }
    let header_length = u16::from_le_bytes([bytes[6], bytes[7]]);
    if header_length as usize != IVF_FILE_HEADER_LEN {
        return Err(IvfReadError::UnsupportedHeaderLength(header_length));
    }
    let mut fourcc = [0u8; 4];
    fourcc.copy_from_slice(&bytes[8..12]);
    let width = u16::from_le_bytes([bytes[12], bytes[13]]);
    let height = u16::from_le_bytes([bytes[14], bytes[15]]);
    let fps_num = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let fps_den = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    let frame_count = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    Ok(IvfFileHeader {
        version,
        header_length,
        fourcc,
        width,
        height,
        fps_num,
        fps_den,
        frame_count,
    })
}

/// IVF v0 demuxer over an in-memory buffer.
///
/// Reads the 32-byte file header on construction and exposes an
/// iterator over the per-frame `(pts, payload)` records that follow.
/// The reader retains a borrow of the underlying byte slice; the
/// produced [`IvfFrame`] payloads are independent `Vec<u8>` copies so
/// callers can drop the source buffer once iteration ends.
#[derive(Debug)]
pub struct IvfReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    header: IvfFileHeader,
}

impl<'a> IvfReader<'a> {
    /// Parse the file header and position the reader at the first
    /// per-frame record.
    pub fn new(bytes: &'a [u8]) -> Result<Self, IvfReadError> {
        let header = parse_file_header(bytes)?;
        Ok(Self {
            bytes,
            cursor: IVF_FILE_HEADER_LEN,
            header,
        })
    }

    /// Parsed file header.
    pub fn header(&self) -> &IvfFileHeader {
        &self.header
    }

    /// Read the next frame's (pts, payload) pair. Returns `Ok(None)`
    /// when the buffer is exhausted at a frame boundary.
    pub fn read_next_frame(&mut self) -> Result<Option<IvfFrame>, IvfReadError> {
        if self.cursor >= self.bytes.len() {
            return Ok(None);
        }
        if self.cursor + IVF_FRAME_HEADER_LEN > self.bytes.len() {
            return Err(IvfReadError::UnexpectedEnd);
        }
        let size = u32::from_le_bytes([
            self.bytes[self.cursor],
            self.bytes[self.cursor + 1],
            self.bytes[self.cursor + 2],
            self.bytes[self.cursor + 3],
        ]) as usize;
        let pts = u64::from_le_bytes([
            self.bytes[self.cursor + 4],
            self.bytes[self.cursor + 5],
            self.bytes[self.cursor + 6],
            self.bytes[self.cursor + 7],
            self.bytes[self.cursor + 8],
            self.bytes[self.cursor + 9],
            self.bytes[self.cursor + 10],
            self.bytes[self.cursor + 11],
        ]);
        let payload_start = self.cursor + IVF_FRAME_HEADER_LEN;
        let payload_end = payload_start
            .checked_add(size)
            .ok_or(IvfReadError::UnexpectedEnd)?;
        if payload_end > self.bytes.len() {
            return Err(IvfReadError::UnexpectedEnd);
        }
        let payload = self.bytes[payload_start..payload_end].to_vec();
        self.cursor = payload_end;
        Ok(Some(IvfFrame { pts, payload }))
    }

    /// Drain every remaining frame into a `Vec<IvfFrame>`. Returns the
    /// first read error encountered.
    pub fn read_all(mut self) -> Result<Vec<IvfFrame>, IvfReadError> {
        let mut out = Vec::new();
        while let Some(frame) = self.read_next_frame()? {
            out.push(frame);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn file_header_byte_exact() {
        let h = build_file_header(FOURCC_AV01, 16, 16, 25, 1, 1);
        assert_eq!(h.len(), 32);
        // Compare byte-for-byte against the
        // `tiny-i-only-16x16-prof0/input.ivf` fixture's first 32 bytes,
        // observed via `xxd`:
        //   DKIF .. 20 00  AV01  10 00 10 00  19 00 00 00  01 00 00 00
        //   01 00 00 00  00 00 00 00
        // (version=0x0000, header_len=0x0020, AV01, 16x16, fps_num=25,
        // fps_den=1, frame_count=1, unused=0.)
        let expected = [
            0x44, 0x4B, 0x49, 0x46, // DKIF
            0x00, 0x00, // version = 0
            0x20, 0x00, // header_length = 32
            0x41, 0x56, 0x30, 0x31, // AV01
            0x10, 0x00, // width = 16
            0x10, 0x00, // height = 16
            0x19, 0x00, 0x00, 0x00, // fps_num = 25
            0x01, 0x00, 0x00, 0x00, // fps_den = 1
            0x01, 0x00, 0x00, 0x00, // frame_count = 1
            0x00, 0x00, 0x00, 0x00, // unused
        ];
        assert_eq!(h, expected);
    }

    #[test]
    fn frame_header_byte_exact() {
        let h = build_frame_header(0x24, 0);
        // Same as the fixture's first frame header:
        //   24 00 00 00 00 00 00 00 00 00 00 00.
        assert_eq!(h, [0x24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn new_writes_file_header_with_zero_frame_count() {
        let mut buf = Vec::new();
        {
            let _writer = IvfWriter::new(&mut buf, FOURCC_AV01, 320, 240, 30, 1).unwrap();
        }
        assert_eq!(buf.len(), 32);
        assert_eq!(&buf[0..4], b"DKIF");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), 0);
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 32);
        assert_eq!(&buf[8..12], b"AV01");
        assert_eq!(u16::from_le_bytes([buf[12], buf[13]]), 320);
        assert_eq!(u16::from_le_bytes([buf[14], buf[15]]), 240);
        assert_eq!(u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]), 30);
        assert_eq!(u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]), 1);
        assert_eq!(u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]), 0);
    }

    #[test]
    fn single_frame_write_layout() {
        let mut buf = Vec::new();
        let payload = [0xAA, 0xBB, 0xCC, 0xDD];
        {
            let mut w = IvfWriter::new(&mut buf, FOURCC_AV01, 16, 16, 25, 1).unwrap();
            w.write_frame(&payload, 0).unwrap();
            assert_eq!(w.frames_written(), 1);
        }
        // 32 (file header) + 12 (frame header) + 4 (payload) = 48.
        assert_eq!(buf.len(), 48);
        // Frame header begins at offset 32.
        assert_eq!(u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]), 4);
        assert_eq!(
            u64::from_le_bytes([
                buf[36], buf[37], buf[38], buf[39], buf[40], buf[41], buf[42], buf[43]
            ]),
            0
        );
        assert_eq!(&buf[44..48], &payload);
    }

    #[test]
    fn multi_frame_layout_and_pts_round_trip() {
        let mut buf = Vec::new();
        let frames: &[(&[u8], u64)] = &[(&[0x01, 0x02], 0), (&[0x03], 1), (&[0x04, 0x05, 0x06], 5)];
        {
            let mut w = IvfWriter::new(&mut buf, FOURCC_AV01, 16, 16, 30, 1).unwrap();
            for (payload, pts) in frames {
                w.write_frame(payload, *pts).unwrap();
            }
            assert_eq!(w.frames_written(), 3);
        }
        let mut cursor = 32usize;
        for (payload, pts) in frames {
            let size = u32::from_le_bytes([
                buf[cursor],
                buf[cursor + 1],
                buf[cursor + 2],
                buf[cursor + 3],
            ]) as usize;
            assert_eq!(size, payload.len());
            let read_pts = u64::from_le_bytes([
                buf[cursor + 4],
                buf[cursor + 5],
                buf[cursor + 6],
                buf[cursor + 7],
                buf[cursor + 8],
                buf[cursor + 9],
                buf[cursor + 10],
                buf[cursor + 11],
            ]);
            assert_eq!(read_pts, *pts);
            assert_eq!(&buf[cursor + 12..cursor + 12 + size], *payload);
            cursor += 12 + size;
        }
        assert_eq!(cursor, buf.len());
    }

    #[test]
    fn patch_frame_count_updates_file_header() {
        let backing = Vec::new();
        let mut w = IvfWriter::new(Cursor::new(backing), FOURCC_AV01, 16, 16, 25, 1).unwrap();
        w.write_frame(&[0xDE, 0xAD], 0).unwrap();
        w.write_frame(&[0xBE, 0xEF], 1).unwrap();
        w.patch_frame_count().unwrap();
        let cursor = w.into_inner();
        let buf = cursor.into_inner();
        assert_eq!(u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]), 2);
        // Frames still present after the seek-back.
        assert_eq!(buf.len(), 32 + (12 + 2) * 2);
    }

    #[test]
    fn empty_payload_frame_is_legal() {
        // 12 bytes of frame header, zero payload.
        let mut buf = Vec::new();
        {
            let mut w = IvfWriter::new(&mut buf, FOURCC_AV01, 16, 16, 25, 1).unwrap();
            w.write_frame(&[], 42).unwrap();
        }
        assert_eq!(buf.len(), 32 + 12);
        assert_eq!(u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]), 0);
        assert_eq!(
            u64::from_le_bytes([
                buf[36], buf[37], buf[38], buf[39], buf[40], buf[41], buf[42], buf[43]
            ]),
            42
        );
    }

    #[test]
    fn fourcc_constants_round_trip_in_file_header() {
        for fourcc in [FOURCC_AV01, FOURCC_VP80, FOURCC_VP90] {
            let h = build_file_header(fourcc, 8, 8, 1, 1, 0);
            assert_eq!(h[8..12], fourcc);
        }
    }

    #[test]
    fn into_inner_returns_sink() {
        let buf = Vec::new();
        let w = IvfWriter::new(buf, FOURCC_AV01, 16, 16, 25, 1).unwrap();
        let inner = w.into_inner();
        assert_eq!(inner.len(), 32);
    }

    // --- IVF v0 reader (round 224) -------------------------------------

    #[test]
    fn parse_file_header_round_trip() {
        let built = build_file_header(FOURCC_AV01, 320, 240, 30, 1, 7);
        let parsed = parse_file_header(&built).expect("file header parses");
        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.header_length as usize, IVF_FILE_HEADER_LEN);
        assert_eq!(parsed.fourcc, FOURCC_AV01);
        assert_eq!(parsed.width, 320);
        assert_eq!(parsed.height, 240);
        assert_eq!(parsed.fps_num, 30);
        assert_eq!(parsed.fps_den, 1);
        assert_eq!(parsed.frame_count, 7);
    }

    #[test]
    fn parse_file_header_rejects_short_buffer() {
        let err = parse_file_header(&[0u8; 16]).unwrap_err();
        assert_eq!(err, IvfReadError::UnexpectedEnd);
    }

    #[test]
    fn parse_file_header_rejects_bad_magic() {
        let mut h = build_file_header(FOURCC_AV01, 16, 16, 25, 1, 0);
        h[0] = b'X';
        let err = parse_file_header(&h).unwrap_err();
        assert_eq!(err, IvfReadError::BadMagic);
    }

    #[test]
    fn ivf_reader_round_trips_writer_output() {
        // Round-trip: writer → reader, three variable-length frames.
        let mut backing = Vec::new();
        {
            let mut w =
                IvfWriter::new(Cursor::new(&mut backing), FOURCC_AV01, 16, 16, 25, 1).unwrap();
            w.write_frame(&[0xAA, 0xBB], 0).unwrap();
            w.write_frame(&[0xCC], 5).unwrap();
            w.write_frame(&[0xDD, 0xEE, 0xFF, 0x00, 0x11], 42).unwrap();
            w.patch_frame_count().unwrap();
        }
        let r = IvfReader::new(&backing).unwrap();
        assert_eq!(r.header().fourcc, FOURCC_AV01);
        assert_eq!(r.header().width, 16);
        assert_eq!(r.header().height, 16);
        assert_eq!(r.header().frame_count, 3);
        let frames = r.read_all().unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].pts, 0);
        assert_eq!(frames[0].payload, vec![0xAA, 0xBB]);
        assert_eq!(frames[1].pts, 5);
        assert_eq!(frames[1].payload, vec![0xCC]);
        assert_eq!(frames[2].pts, 42);
        assert_eq!(frames[2].payload, vec![0xDD, 0xEE, 0xFF, 0x00, 0x11]);
    }

    #[test]
    fn ivf_reader_empty_buffer_returns_none() {
        let mut backing = Vec::new();
        {
            let _ = IvfWriter::new(&mut backing, FOURCC_AV01, 16, 16, 25, 1).unwrap();
        }
        // Just a file header, no frames.
        let mut r = IvfReader::new(&backing).unwrap();
        assert!(r.read_next_frame().unwrap().is_none());
    }

    #[test]
    fn ivf_reader_truncated_frame_header_errors() {
        // 32-byte header + only 6 bytes of a 12-byte frame header.
        let mut buf = build_file_header(FOURCC_AV01, 16, 16, 25, 1, 1).to_vec();
        buf.extend_from_slice(&[0; 6]);
        let mut r = IvfReader::new(&buf).unwrap();
        assert_eq!(
            r.read_next_frame().unwrap_err(),
            IvfReadError::UnexpectedEnd
        );
    }

    #[test]
    fn ivf_reader_truncated_payload_errors() {
        // Frame header claims 10-byte payload, only 4 supplied.
        let mut buf = build_file_header(FOURCC_AV01, 16, 16, 25, 1, 1).to_vec();
        buf.extend_from_slice(&build_frame_header(10, 0));
        buf.extend_from_slice(&[0u8; 4]);
        let mut r = IvfReader::new(&buf).unwrap();
        assert_eq!(
            r.read_next_frame().unwrap_err(),
            IvfReadError::UnexpectedEnd
        );
    }
}
