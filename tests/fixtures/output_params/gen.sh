#!/bin/sh
# Regenerates the tests/output_params.rs fixtures with black-box ffmpeg
# (libsvtav1 encoder) and prints what ffprobe reports for each, which
# is what the test pins. Run from this directory.
set -e
enc() {
    out=$1; shift
    ffmpeg -hide_banner -loglevel error -y "$@" -c:v libsvtav1 -preset 12 -crf 50 "$out"
    printf '%s: ' "$out"
    ffprobe -v error -show_streams -of compact "$out" | tr '|' '\n' |
        grep -E '^(width|height|pix_fmt|r_frame_rate|color_range|color_space|color_transfer|color_primaries)=' |
        tr '\n' ' '
    echo
}
# 8-bit 4:2:0, a size that is not a multiple of 8, limited range.
enc svt_420p_70x46.ivf -f lavfi -i testsrc=size=70x46:rate=30 -frames:v 2 -pix_fmt yuv420p
# 10-bit 4:2:0, BT.709 colour description, full range.
enc svt_420p10_64x48_bt709_full.ivf -f lavfi -i testsrc=size=64x48:rate=24 -frames:v 2 \
    -pix_fmt yuv420p10le \
    -svtav1-params color-primaries=1:transfer-characteristics=1:matrix-coefficients=1:color-range=1
# The 4:2:2 / 4:4:4 / monochrome / 12-bit layouts libsvtav1 cannot
# produce are encoded in-process by tests/output_params.rs with this
# crate's own encoder; setting OXIDEAV_AV1_OUTPUT_PARAMS_DUMP=<dir>
# writes those streams out as IVF for the same ffprobe check.
