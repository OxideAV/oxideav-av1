//! The registry encoder defaults to 8-bit 4:2:0 when the stream
//! parameters carry no pixel format.

use oxideav_core::{CodecId, CodecParameters, PixelFormat};

#[test]
fn encoder_without_pixel_format_defaults_to_yuv420p() {
    let mut p = CodecParameters::video(CodecId::new("av1"));
    p.width = Some(64);
    p.height = Some(64);
    let enc = oxideav_av1::registry::make_encoder(&p).expect("encoder without pixel_format");
    assert_eq!(enc.output_params().pixel_format, Some(PixelFormat::Yuv420P));
}
