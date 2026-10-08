#!/bin/sh
# A2 step 1 for every frozen encoder (hjepa/encoder_probe.py): LeVJEPA-L and V-JEPA 2.1 L, g and G on the
# 224x224 val set (data/so101_val_224.h5, sim/make_dataset.sh). Fetches what is missing first: the
# V-JEPA 2.1 weights from dl.fbaipublicfiles.com into data/weights/ (5.2, 16.9 and 30.2 GB) and the
# vjepa2 model code at a pinned commit into data/vjepa2_src/. One encoder failing does not stop the rest.
# Usage: sh hjepa/a2_encoders.sh [ENCODERS]   (default "levjepa vjepa21_L vjepa21_g vjepa21_G")
# Results: outputs/a2_probe_ENCODER.json and .log
. "$(dirname "$0")/../scripts/gpu.sh"
ENCODERS=${1:-levjepa vjepa21_L vjepa21_g vjepa21_G}
SHA=204698b45b3712590f06245fbfba32d3be539812  # facebookresearch/vjepa2 main, 2026-10-08
URL=https://dl.fbaipublicfiles.com/vjepa2
mkdir -p data/weights outputs
if [ ! -f data/vjepa2_src/.sha ] || [ "$(cat data/vjepa2_src/.sha)" != $SHA ]; then
  rm -rf data/vjepa2_src && mkdir -p data/vjepa2_src
  curl -sSL https://codeload.github.com/facebookresearch/vjepa2/tar.gz/$SHA | tar -xz -C data/vjepa2_src --strip-components 1 &&
    echo $SHA > data/vjepa2_src/.sha || { echo "a2_encoders: could not fetch the vjepa2 code" >&2; exit 1; }
fi
fetch() {  # file expected-bytes
  [ "$(stat -c %s "data/weights/$1" 2> /dev/null)" = "$2" ] && return 0
  echo "a2_encoders: downloading $1" >&2
  curl -sSL --retry 5 -C - -o "data/weights/$1" "$URL/$1" && [ "$(stat -c %s "data/weights/$1")" = "$2" ]
}
fail=0
for e in $ENCODERS; do
  case $e in
    vjepa21_L) fetch vjepa2_1_vitl_dist_vitG_384.pt 5151198524 || { fail=1; continue; } ;;
    vjepa21_g) fetch vjepa2_1_vitg_384.pt 16878318788 || { fail=1; continue; } ;;
    vjepa21_G) fetch vjepa2_1_vitG_384.pt 30238058912 || { fail=1; continue; } ;;
  esac
  echo "a2_encoders: $e" >&2
  uvr python hjepa/encoder_probe.py data/so101_val_224.h5 outputs/a2_probe_$e.json $e \
    > outputs/a2_probe_$e.log 2>&1 || { fail=1; echo "a2_encoders: $e failed (outputs/a2_probe_$e.log)" >&2; }
done
exit $fail
