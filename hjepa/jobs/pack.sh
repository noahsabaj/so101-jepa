#!/bin/sh
# A small copy of what the LeWAM jobs need (LeWAM's code, our wrappers, the jobs), for fleet's Colab (20 MB) and Kaggle
# nodes, which pack the whole project folder:  sh hjepa/jobs/pack.sh DEST; cd DEST; fleet submit --on kaggle ...
set -eu
D=$1
mkdir -p "$D/hjepa/config" "$D/third_party"
rm -rf "$D/third_party/lewam" "$D/hjepa/config/lewam" "$D/hjepa/jobs"
cp -r third_party/lewam "$D/third_party/"
rm -rf "$D/third_party/lewam/.git" "$D/third_party/lewam/assets"
cp -r hjepa/config/lewam "$D/hjepa/config/"
cp -r hjepa/jobs "$D/hjepa/"
cp hjepa/lewam_common.py hjepa/lewam_train.py hjepa/lewam_decompress.py hjepa/lewam_repro_env.sh hjepa/lewam_repro_node.sh \
    hjepa/lewam_repro_table.py "$D/hjepa/"
du -sh "$D"
