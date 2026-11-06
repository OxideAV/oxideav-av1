#!/bin/sh
# Regenerates the tests/stream_params.rs fixtures with black-box
# encoders and prints what the test pins: per-frame sizes from
# `ffprobe -show_frames`, and the FNV-1a 64 digest of the reference
# output (`aomdec --rawvideo`, also byte-identical to `dav1d -o *.yuv`).
# Run from this directory.
set -e
ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc=size=96x64:rate=30 \
    -frames:v 16 -pix_fmt yuv420p src.y4m
AOM="--ivf --limit=16 --cpu-used=8 --end-usage=q --cq-level=50 --lag-in-frames=8"
# Random per-frame resize (§5.9.5 frame_size_override_flag / §5.9.7
# frame_size_with_refs, scaled references).
aomenc $AOM -o aom_resize_random.ivf --auto-alt-ref=1 --resize-mode=2 --kf-max-dist=999 src.y4m
# Random superres only: the upscaled output size never changes.
aomenc $AOM -o aom_superres_random.ivf --superres-mode=2 src.y4m
# Both: superres on resized frames.
aomenc $AOM -o aom_resize_superres.ivf --resize-mode=2 --superres-mode=2 src.y4m
SVT="-i src.y4m -n 16 --preset 12 --crf 50"
# Two scheduled resize events (frames 4 and 10), each landing on a
# show_existing_frame output.
SvtAv1EncApp $SVT -b svt_resize_ra.ivf --resize-mode 4 --frame-resz-events 4,10 \
    --frame-resz-kf-denoms 12,16 --frame-resz-denoms 12,16
# Random per-frame resize under a hierarchical group (show_existing
# outputs of resized frames).
SvtAv1EncApp $SVT -b svt_resize_random.ivf --resize-mode 2
for f in aom_resize_random aom_superres_random aom_resize_superres svt_resize_ra svt_resize_random; do
    printf '%s: ' $f
    ffprobe -v error -show_frames -of csv=p=0 -show_entries frame=width,height $f.ivf | tr '\n' ' '
    aomdec --rawvideo -o $f.yuv $f.ivf
    python3 -c "import sys
h=0xcbf29ce484222325
for b in open(sys.argv[1],'rb').read():
    h=((h^b)*0x100000001b3)&0xffffffffffffffff
print(' digest %016x' % h)" $f.yuv
    rm $f.yuv
done
rm src.y4m
