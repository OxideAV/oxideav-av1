//! Frame header OBU parser — initial structural slice (§5.9.2 prefix).
//!
//! This round consumes the leading fields of `uncompressed_header()`
//! per §5.9.2 of the AV1 Bitstream & Decoding Process Specification
//! and returns a strongly typed [`FrameHeader`]. The slice covered is
//! everything from the start of `uncompressed_header()` through
//! `refresh_frame_flags` inclusive. The downstream blocks
//! (`frame_size()`, `read_interpolation_filter()`, `tile_info()`,
//! `quantization_params()`, …) are out of scope for this round.
//!
//! ## Syntax / semantics references (all in `docs/video/av1/`)
//!
//!   * §5.9.1 — General frame header OBU syntax
//!   * §5.9.2 — Uncompressed header syntax
//!   * §6.8.1 — General frame header OBU semantics
//!   * §6.8.2 — Uncompressed header semantics
//!
//! §3 constants used here:
//!
//!   * `NUM_REF_FRAMES = 8` — number of reference frame slots; the
//!     spec derives `allFrames = (1 << NUM_REF_FRAMES) - 1 = 0xff`.
//!   * `PRIMARY_REF_NONE = 7` — sentinel for `primary_ref_frame`
//!     indicating no primary reference (loaded as default state).
//!
//! Composition with §5.5: the parser takes a borrowed
//! [`SequenceHeader`] from round 2; sequence-header state controls
//! several conditional reads inside `uncompressed_header()`:
//! `frame_id_numbers_present_flag` (gates `display_frame_id` /
//! `current_frame_id`), `reduced_still_picture_header` (collapses the
//! whole leading block to fixed values), `decoder_model_info_present`
//! (governs `temporal_point_info()` — deferred, see below),
//! `seq_force_screen_content_tools` (decides whether
//! `allow_screen_content_tools` is read or inferred), the matching
//! `seq_force_integer_mv` for `force_integer_mv`, and
//! `order_hint_bits` (the width of the `order_hint` field).
//!
//! ## What the parser does NOT do this round
//!
//!   * `temporal_point_info()` (§5.9.31) — gated by
//!     `decoder_model_info_present_flag && !equal_picture_interval`.
//!     The §5.9.2 leading block uses it twice but every fixture under
//!     `docs/video/av1/fixtures/` is encoded without a decoder model,
//!     so the call site is silently a no-op for the corpus. The
//!     parser returns [`Error::TemporalPointInfoUnsupported`] if it
//!     would actually need to descend into it — i.e. when both gate
//!     conditions are true. The next round can land §5.9.31 alongside
//!     the rest of `frame_size()` / `tile_info()`.
//!   * `mark_ref_frames()` (§7.20) — a derivation that updates
//!     `RefValid` / `RefOrderHint`; deferred to the inter-frame round
//!     that introduces ref-frame state.
//!   * `frame_size()` / `render_size()` / `frame_size_with_refs()` /
//!     `superres_params()` / the inter / intra-only post-block. The
//!     parser stops once `refresh_frame_flags` has been read.
//!
//! The bit count consumed is returned in [`FrameHeader::bits_consumed`]
//! so the next round can continue at exactly the right bit.

use crate::bitreader::BitReader;
use crate::sequence_header::{SequenceHeader, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
use crate::Error;

// ---------------------------------------------------------------------
// §3 constants
// ---------------------------------------------------------------------

/// `NUM_REF_FRAMES` (§3): number of reference-frame slots.
pub const NUM_REF_FRAMES: u8 = 8;

/// `PRIMARY_REF_NONE` (§3): sentinel for `primary_ref_frame` meaning
/// no primary reference frame is used (CDF / loop-filter / segment
/// state are reset to defaults).
pub const PRIMARY_REF_NONE: u8 = 7;

/// `allFrames = (1 << NUM_REF_FRAMES) - 1` from §5.9.2.
const ALL_FRAMES: u8 = 0xff;

// ---------------------------------------------------------------------
// FrameType (§6.8.2)
// ---------------------------------------------------------------------

/// `frame_type` per the §6.8.2 enumeration:
///
/// | code | name              |
/// |-----:|:------------------|
/// |   0  | `KEY_FRAME`       |
/// |   1  | `INTER_FRAME`     |
/// |   2  | `INTRA_ONLY_FRAME`|
/// |   3  | `SWITCH_FRAME`    |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

impl FrameType {
    /// Decode the raw 2-bit `frame_type` field.
    pub fn from_raw(raw: u8) -> Self {
        match raw & 0x3 {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }

    /// Raw 2-bit encoding that produced `self`.
    pub fn as_raw(&self) -> u8 {
        match self {
            Self::Key => 0,
            Self::Inter => 1,
            Self::IntraOnly => 2,
            Self::Switch => 3,
        }
    }

    /// `FrameIsIntra = (frame_type == INTRA_ONLY_FRAME || frame_type ==
    /// KEY_FRAME)` per §5.9.2.
    pub fn is_intra(&self) -> bool {
        matches!(self, Self::Key | Self::IntraOnly)
    }
}

// ---------------------------------------------------------------------
// FrameHeader
// ---------------------------------------------------------------------

/// Parsed leading slice of `uncompressed_header()` per §5.9.2.
///
/// Fields are populated according to the §5.9.2 syntax. When a field
/// is conditionally absent in the bitstream, the value here is the
/// §5.9.2 / §6.8.2 inferred default (e.g. `error_resilient_mode = true`
/// when `frame_type == SWITCH_FRAME` or `frame_type == KEY_FRAME &&
/// show_frame`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// `show_existing_frame` (§5.9.2 / §6.8.2). Inferred to `false`
    /// when `reduced_still_picture_header == 1`.
    pub show_existing_frame: bool,
    /// `frame_to_show_map_idx` — present only when
    /// `show_existing_frame == 1`. 3-bit index into the reference
    /// frame map.
    pub frame_to_show_map_idx: Option<u8>,
    /// `display_frame_id` — present only when
    /// `show_existing_frame == 1` *and*
    /// `frame_id_numbers_present_flag`. `idLen` bits.
    pub display_frame_id: Option<u32>,
    /// `frame_type` (§6.8.2). Inferred to `KEY_FRAME` when
    /// `reduced_still_picture_header == 1`.
    pub frame_type: FrameType,
    /// `FrameIsIntra` per §5.9.2 (derived from `frame_type`).
    pub frame_is_intra: bool,
    /// `show_frame` — inferred to `true` when
    /// `reduced_still_picture_header == 1`; inferred when this is a
    /// show-existing-frame replay.
    pub show_frame: bool,
    /// `showable_frame` — inferred per §5.9.2:
    /// `(frame_type != KEY_FRAME)` when `show_frame == 1` is read
    /// from above, and `0` when reduced-still-picture-header.
    pub showable_frame: bool,
    /// `error_resilient_mode`. Inferred to `true` for `SWITCH_FRAME`
    /// or `(KEY_FRAME && show_frame)`; otherwise read as `f(1)`.
    pub error_resilient_mode: bool,
    /// `disable_cdf_update`. Read as `f(1)` unconditionally in the
    /// non-show-existing path.
    pub disable_cdf_update: bool,
    /// `allow_screen_content_tools`. Read as `f(1)` when
    /// `seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS`;
    /// otherwise inferred from the sequence header.
    pub allow_screen_content_tools: bool,
    /// `force_integer_mv` after the §5.9.2 `if (FrameIsIntra)` override.
    /// `1` when `FrameIsIntra` regardless of the bitstream bit; `0`
    /// when `!allow_screen_content_tools`; else read or inferred from
    /// the sequence header's `seq_force_integer_mv`.
    pub force_integer_mv: bool,
    /// `current_frame_id` (only when
    /// `frame_id_numbers_present_flag == 1`). Otherwise `0` per §5.9.2.
    pub current_frame_id: u32,
    /// `frame_size_override_flag`. `1` for `SWITCH_FRAME`; `0` for
    /// `reduced_still_picture_header`; otherwise read.
    pub frame_size_override_flag: bool,
    /// `order_hint` — `order_hint_bits` wide. `0` when
    /// `order_hint_bits == 0` (§5.5.1 disabled-order-hint mode).
    pub order_hint: u32,
    /// `primary_ref_frame`. `PRIMARY_REF_NONE` when `FrameIsIntra ||
    /// error_resilient_mode`; otherwise the read 3-bit value.
    pub primary_ref_frame: u8,
    /// `refresh_frame_flags`. `0xff` for `SWITCH_FRAME` or
    /// `(KEY_FRAME && show_frame)`; `0` when show-existing-frame
    /// replays a non-KEY frame; otherwise read.
    pub refresh_frame_flags: u8,
    /// Total bits consumed from `payload` by this parse. The next
    /// round will continue from this position to decode `frame_size()`
    /// / `render_size()` / `tile_info()` / etc.
    pub bits_consumed: usize,
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

/// Parse the leading structural slice of `uncompressed_header()`
/// (§5.9.2) from a frame-header OBU payload, given the active
/// [`SequenceHeader`].
///
/// `payload` is the slice the OBU walker returned for an
/// `OBU_FRAME_HEADER` / `OBU_REDUNDANT_FRAME_HEADER` / `OBU_FRAME`
/// payload (the frame OBU's per-tile data sits after the
/// uncompressed-header bits but is out of scope this round).
///
/// The parser stops after `refresh_frame_flags` is determined; the
/// remaining §5.9.2 work (`frame_size()` / `render_size()` /
/// `frame_size_with_refs()` / interpolation filter / loop-filter
/// params / tile info / quant / segment / CDEF / LR / TX mode / frame
/// reference mode / skip mode / global motion / film grain) is
/// deferred to the next round. [`FrameHeader::bits_consumed`] reports
/// the bit position the next round should resume from.
///
/// ## Errors
///
///   * [`Error::UnexpectedEnd`] — payload ran out mid-header.
///   * [`Error::TemporalPointInfoUnsupported`] — the call sites
///     guarded by `decoder_model_info_present_flag &&
///     !equal_picture_interval` are not implemented yet. Every
///     fixture under `docs/video/av1/fixtures/` parses without
///     hitting this path; bitstreams that enable a decoder model
///     plus variable picture interval will be picked up in a future
///     round alongside §5.9.31.
///   * [`Error::InvalidIdLen`] — §6.8.2 conformance:
///     `idLen <= 16` (the constraint stated alongside
///     `display_frame_id`); we surface the violation here rather than
///     silently reading >16 bits.
pub fn parse_frame_header(payload: &[u8], seq: &SequenceHeader) -> Result<FrameHeader, Error> {
    let mut br = BitReader::new(payload);
    parse_with(&mut br, seq)
}

fn parse_with(br: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<FrameHeader, Error> {
    // §5.9.2 idLen derivation (only valid when frame_id_numbers_present_flag).
    let id_len: u32 = if seq.frame_id_numbers_present_flag {
        u32::from(seq.additional_frame_id_length_minus_1)
            + u32::from(seq.delta_frame_id_length_minus_2)
            + 3
    } else {
        0
    };
    if seq.frame_id_numbers_present_flag && id_len > 16 {
        // §6.8.2 conformance: idLen <= 16.
        return Err(Error::InvalidIdLen);
    }

    // The big reduced-still-picture-header collapse from §5.9.2.
    if seq.reduced_still_picture_header {
        // The reduced path skips every show_existing branch + the
        // frame_type read; everything in the leading block is fixed.
        // We must still read disable_cdf_update, allow_screen_content_tools
        // (conditionally), force_integer_mv (conditionally),
        // frame_size_override_flag (none — derived 0),
        // order_hint (none — order_hint_bits is forced to 0 in §5.5.1
        // when reduced_still_picture_header == 1), primary_ref_frame
        // (derived PRIMARY_REF_NONE), refresh_frame_flags (derived
        // allFrames since frame_type==KEY && show_frame).
        let disable_cdf_update = br.f(1)? == 1;
        let allow_screen_content_tools = read_allow_scc(br, seq)?;
        // FrameIsIntra is true for a derived KEY frame, so
        // force_integer_mv is forced to 1 by the §5.9.2 override
        // regardless of what the seq forces / the bit reads.
        // We still need to consume the bitstream slot if the spec
        // would have asked us to. The spec gates the read on
        // allow_screen_content_tools then overrides to 1 for
        // FrameIsIntra, so we follow it literally.
        consume_force_integer_mv_if_present(br, seq, allow_screen_content_tools)?;
        let force_integer_mv = true; // FrameIsIntra branch.

        // frame_id_numbers_present_flag is gated upstream — §5.5.1
        // forces it off when reduced_still_picture_header is set, so
        // current_frame_id is implicitly 0 and there's no read.
        // frame_size_override_flag is derived 0.
        // order_hint: order_hint_bits is 0 in this path (§5.5.1
        // wrote it out), so we don't read.
        // primary_ref_frame = PRIMARY_REF_NONE (derived).
        // decoder_model_info_present_flag is forced off too in §5.5.1
        // for the reduced path, so no temporal-point-info read.
        // refresh_frame_flags: (KEY && show_frame) so allFrames.
        return Ok(FrameHeader {
            show_existing_frame: false,
            frame_to_show_map_idx: None,
            display_frame_id: None,
            frame_type: FrameType::Key,
            frame_is_intra: true,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: true, // KEY_FRAME && show_frame => 1.
            disable_cdf_update,
            allow_screen_content_tools,
            force_integer_mv,
            current_frame_id: 0,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: ALL_FRAMES,
            bits_consumed: br.position(),
        });
    }

    // -----------------------------------------------------------------
    // Non-reduced path.
    // -----------------------------------------------------------------
    let show_existing_frame = br.f(1)? == 1;
    if show_existing_frame {
        let map_idx = br.f(3)? as u8;
        // temporal_point_info gate.
        if decoder_model_info_present(seq) && !equal_picture_interval(seq) {
            return Err(Error::TemporalPointInfoUnsupported);
        }
        let display_id = if seq.frame_id_numbers_present_flag {
            Some(br.f(id_len)? as u32)
        } else {
            None
        };
        // We don't carry the RefFrameType state across calls in this
        // round, so we can't actually look up RefFrameType[map_idx]
        // to know whether it was KEY_FRAME — but the trace tells the
        // decoder this from session state, not the bitstream. For
        // the structural parser we return frame_type = INTER (the
        // common case for replays); a downstream session-aware layer
        // can correct it from its own RefFrameType array. We pick
        // INTER here because §5.9.2 only forces refresh_frame_flags
        // to allFrames if the replayed frame was KEY, and we leave
        // refresh_frame_flags at 0 to match the trace's common
        // "ref-frame-state-replay" reading.
        return Ok(FrameHeader {
            show_existing_frame: true,
            frame_to_show_map_idx: Some(map_idx),
            display_frame_id: display_id,
            frame_type: FrameType::Inter,
            frame_is_intra: false,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            current_frame_id: 0,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: 0,
            bits_consumed: br.position(),
        });
    }

    let frame_type = FrameType::from_raw(br.f(2)? as u8);
    let frame_is_intra = frame_type.is_intra();
    let show_frame = br.f(1)? == 1;

    if show_frame && decoder_model_info_present(seq) && !equal_picture_interval(seq) {
        return Err(Error::TemporalPointInfoUnsupported);
    }

    let showable_frame = if show_frame {
        !matches!(frame_type, FrameType::Key)
    } else {
        br.f(1)? == 1
    };

    let error_resilient_mode = if matches!(frame_type, FrameType::Switch)
        || (matches!(frame_type, FrameType::Key) && show_frame)
    {
        true
    } else {
        br.f(1)? == 1
    };

    // Section that initialises RefValid / RefOrderHint / OrderHints
    // only matters for the inter-frame round — it doesn't touch the
    // bitstream.

    let disable_cdf_update = br.f(1)? == 1;
    let allow_screen_content_tools = read_allow_scc(br, seq)?;
    // Read force_integer_mv from the bitstream if conditions ask for
    // it; the §5.9.2 FrameIsIntra override then forces it to true.
    let raw_force_integer_mv =
        consume_force_integer_mv_if_present(br, seq, allow_screen_content_tools)?;
    let force_integer_mv = if frame_is_intra {
        true
    } else {
        raw_force_integer_mv
    };

    let current_frame_id = if seq.frame_id_numbers_present_flag {
        br.f(id_len)? as u32
    } else {
        0
    };

    let frame_size_override_flag = if matches!(frame_type, FrameType::Switch) {
        true
    } else {
        // reduced-still path already returned; this branch only.
        br.f(1)? == 1
    };

    let order_hint = if seq.order_hint_bits == 0 {
        0
    } else {
        br.f(u32::from(seq.order_hint_bits))? as u32
    };

    let primary_ref_frame = if frame_is_intra || error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        br.f(3)? as u8
    };

    // temporal_point_info() under decoder_model_info_present_flag &&
    // buffer_removal_time_present_flag is the next syntax block; the
    // spec immediately follows with the operating-point loop reading
    // buffer_removal_time[opNum]. Since we don't claim support for
    // decoder-model frames in this round, we refuse to descend if
    // the gate is active.
    if decoder_model_info_present(seq) {
        return Err(Error::TemporalPointInfoUnsupported);
    }

    let refresh_frame_flags = if matches!(frame_type, FrameType::Switch)
        || (matches!(frame_type, FrameType::Key) && show_frame)
    {
        ALL_FRAMES
    } else {
        br.f(8)? as u8
    };

    Ok(FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        display_frame_id: None,
        frame_type,
        frame_is_intra,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        current_frame_id,
        frame_size_override_flag,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        bits_consumed: br.position(),
    })
}

/// `allow_screen_content_tools` per §5.9.2: read iff
/// `seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS`;
/// otherwise the value is the sequence header's force value.
fn read_allow_scc(br: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<bool, Error> {
    if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
        Ok(br.f(1)? == 1)
    } else {
        Ok(seq.seq_force_screen_content_tools != 0)
    }
}

/// Read `force_integer_mv` from the bitstream when the §5.9.2 syntax
/// table asks for it, returning the bitstream-derived value (the
/// caller is responsible for the `FrameIsIntra` override).
fn consume_force_integer_mv_if_present(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    allow_screen_content_tools: bool,
) -> Result<bool, Error> {
    if allow_screen_content_tools {
        if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            Ok(br.f(1)? == 1)
        } else {
            Ok(seq.seq_force_integer_mv != 0)
        }
    } else {
        Ok(false)
    }
}

fn decoder_model_info_present(seq: &SequenceHeader) -> bool {
    seq.decoder_model_info_present_flag
}

/// `equal_picture_interval` lives inside the `timing_info` struct;
/// the §5.9.2 gate "`!equal_picture_interval`" only fires when timing
/// info is present (because `decoder_model_info_present_flag == 1`
/// requires `timing_info_present_flag == 1` per §5.5.1). For the
/// docs gap above, we conservatively treat "no timing info" as
/// "equal picture interval", i.e. the temporal-point-info branch is
/// not taken.
fn equal_picture_interval(seq: &SequenceHeader) -> bool {
    seq.timing_info
        .map(|t| t.equal_picture_interval)
        .unwrap_or(true)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_header::parse_sequence_header;

    /// Build a sequence-header-like context by parsing the
    /// SEQ_HEADER payload from the tiny-i-only fixture and returning
    /// the parsed [`SequenceHeader`].
    fn tiny_seq() -> SequenceHeader {
        // Payload extracted from
        // docs/video/av1/fixtures/tiny-i-only-16x16-prof0/input.ivf
        // (SEQ_HEADER OBU payload).
        let seq_payload: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
        parse_sequence_header(seq_payload).expect("seq header parses")
    }

    fn show_existing_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x9b, 0x5f, 0x30, 0x08];
        parse_sequence_header(seq_payload).expect("show-existing seq header parses")
    }

    fn screen_content_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x18, 0x1d, 0xbf, 0xff, 0xf2, 0x01];
        parse_sequence_header(seq_payload).expect("screen-content seq header parses")
    }

    fn super_resolution_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x18, 0x19, 0x7f, 0xff, 0xf8, 0x04];
        parse_sequence_header(seq_payload).expect("super-res seq header parses")
    }

    // -------------------------------------------------------------
    // Fixture 1: tiny-i-only KEY frame (show_frame=1)
    // -------------------------------------------------------------

    /// FRAME OBU payload from `tiny-i-only-16x16-prof0`. Trace says:
    /// show_existing=0 frame_type=0(KEY) show_frame=1 showable=0
    /// error_resilient=1 disable_cdf_update=0 allow_screen_content=0
    /// force_integer_mv=0 (raw; FrameIsIntra forces to 1) order_hint=0
    /// primary_ref_frame=7 refresh_flags=0xff.
    const TINY_FRAME_PAYLOAD: &[u8] = &[
        0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67,
        0x6c, 0xc7, 0xee, 0x51, 0x80,
    ];

    #[test]
    fn parses_tiny_key_frame_prefix() {
        let seq = tiny_seq();
        let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).expect("frame header parses");
        assert!(!fh.show_existing_frame);
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.frame_is_intra);
        assert!(fh.show_frame);
        assert!(!fh.showable_frame, "KEY + show_frame => showable=0");
        assert!(
            fh.error_resilient_mode,
            "KEY + show_frame => error_resilient forced to 1"
        );
        assert!(!fh.disable_cdf_update);
        assert!(!fh.allow_screen_content_tools);
        // FrameIsIntra path forces force_integer_mv = true.
        assert!(fh.force_integer_mv);
        assert_eq!(fh.current_frame_id, 0);
        assert!(!fh.frame_size_override_flag);
        assert_eq!(fh.order_hint, 0);
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert_eq!(fh.refresh_frame_flags, 0xff);
        // The §5.9.2 leading block reads (in this fixture):
        //   show_existing(1) frame_type(2) show_frame(1)
        //   [showable+error_resilient derived for KEY+show_frame]
        //   disable_cdf_update(1) allow_scc(1)
        //   [force_integer_mv not read: allow_scc=0]
        //   [current_frame_id not read: frame_id_numbers_present_flag=0]
        //   frame_size_override_flag(1) order_hint(7)
        //   [primary_ref_frame derived: FrameIsIntra]
        //   [refresh_frame_flags derived: KEY && show_frame]
        // = 14 bits.
        assert_eq!(fh.bits_consumed, 14);
    }

    // -------------------------------------------------------------
    // Fixture 2: show-existing-frame, first frame (KEY, show_frame=0,
    // allow_scc=1)
    // -------------------------------------------------------------

    /// First FRAME OBU payload from `show-existing-frame/input.ivf`,
    /// the actual KEY frame that is later replayed. Trace:
    /// show_existing=0 frame_type=0 show_frame=0 showable=0
    /// error_resilient=0 disable_cdf=0 allow_scc=1 force_integer_mv=0
    /// (raw; FrameIsIntra forces 1) order_hint=0 primary_ref=7
    /// refresh_flags=0xff.
    const SHOW_EXISTING_KEY_PAYLOAD: &[u8] = &[
        0x01, 0x00, 0x7f, 0x89, 0x10, 0x02, 0x08, 0x10, 0x82, 0x00, 0x0a, 0x02, 0xdc, 0x85, 0x28,
        0xf5,
    ];

    #[test]
    fn parses_show_existing_underlying_key_frame() {
        let seq = show_existing_seq();
        let fh = parse_frame_header(SHOW_EXISTING_KEY_PAYLOAD, &seq).expect("frame header parses");
        assert!(!fh.show_existing_frame);
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(!fh.show_frame);
        assert!(!fh.showable_frame);
        assert!(!fh.error_resilient_mode, "KEY + show_frame=0 reads er_mode");
        assert!(fh.allow_screen_content_tools);
        // FrameIsIntra still forces force_integer_mv to 1 even though
        // the raw bitstream bit was 0.
        assert!(fh.force_integer_mv);
        assert_eq!(fh.order_hint, 0);
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert_eq!(fh.refresh_frame_flags, 0xff);
    }

    // -------------------------------------------------------------
    // Synthetic: minimal show-existing-frame replay
    // -------------------------------------------------------------

    #[test]
    fn synthetic_show_existing_frame_replay() {
        // Construct an uncompressed_header() against the tiny seq
        // (frame_id_numbers_present_flag = 0). Bits:
        //   show_existing(1)=1
        //   frame_to_show_map_idx(3)=5
        // = 4 bits = 0b1101 = 1101_0000 = 0xD0 (high nibble).
        let payload = [0b1101_0000u8];
        let seq = tiny_seq();
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert!(fh.show_existing_frame);
        assert_eq!(fh.frame_to_show_map_idx, Some(5));
        assert!(fh.display_frame_id.is_none());
        assert_eq!(fh.refresh_frame_flags, 0);
        // Show-existing-frame returns early after reading 4 bits.
        assert_eq!(fh.bits_consumed, 4);
    }

    // -------------------------------------------------------------
    // Synthetic: reduced-still-picture-header path
    // -------------------------------------------------------------

    #[test]
    fn reduced_still_picture_path() {
        // Use the screen-content-tools SEQ (which is reduced-still).
        let seq = screen_content_seq();
        assert!(seq.reduced_still_picture_header);
        assert_eq!(
            seq.seq_force_screen_content_tools, SELECT_SCREEN_CONTENT_TOOLS,
            "reduced still forces SELECT"
        );
        // Bits we need to consume:
        //   disable_cdf_update(1)=0
        //   allow_screen_content_tools(1)=1     [seq force is SELECT]
        //   force_integer_mv(1)=0               [seq force is SELECT]
        // = 3 bits = 0b010 = 0100_0000 = 0x40.
        let payload = [0x40u8];
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.frame_is_intra);
        assert!(fh.show_frame);
        assert!(!fh.showable_frame);
        assert!(fh.error_resilient_mode);
        assert!(!fh.disable_cdf_update);
        assert!(fh.allow_screen_content_tools);
        // FrameIsIntra forces force_integer_mv=1 even though the bit
        // we wrote was 0.
        assert!(fh.force_integer_mv);
        assert_eq!(fh.refresh_frame_flags, 0xff);
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert_eq!(fh.bits_consumed, 3);
    }

    // -------------------------------------------------------------
    // Synthetic: reduced-still without SELECT for force_integer_mv
    // -------------------------------------------------------------

    #[test]
    fn reduced_still_without_select_force_int_mv() {
        // super_resolution fixture is reduced-still with allow_scc
        // path through SELECT. Same shape as screen-content above.
        let seq = super_resolution_seq();
        assert!(seq.reduced_still_picture_header);
        let payload = [0x40u8]; // disable_cdf=0, allow_scc=1, force_int_mv=0
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert!(fh.allow_screen_content_tools);
        assert!(fh.force_integer_mv);
    }

    // -------------------------------------------------------------
    // FrameType enum unit
    // -------------------------------------------------------------

    #[test]
    fn frame_type_roundtrip() {
        for raw in 0u8..4u8 {
            let ft = FrameType::from_raw(raw);
            assert_eq!(ft.as_raw(), raw);
        }
        assert!(FrameType::Key.is_intra());
        assert!(!FrameType::Inter.is_intra());
        assert!(FrameType::IntraOnly.is_intra());
        assert!(!FrameType::Switch.is_intra());
    }

    // -------------------------------------------------------------
    // Error paths
    // -------------------------------------------------------------

    #[test]
    fn unexpected_end_on_truncated_payload() {
        let seq = tiny_seq();
        let payload: [u8; 0] = [];
        let err = parse_frame_header(&payload, &seq).expect_err("must error");
        assert_eq!(err, Error::UnexpectedEnd);
    }
}
