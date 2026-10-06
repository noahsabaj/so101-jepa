#!/bin/sh
# Run a command for this project in WSL on kat-pc (Linux, CUDA), from a fleet job there:
#   fleet run --on kat-pc --gpu-gb 7 -- wsl.exe -d DISTRO -- sh hjepa/wsl_job.sh CMD ARGS...
# The fleet copy (/mnt/c/fleet/so101-jepa) is synced to ext4 (~/so101-jepa), where the venv, data
# and checkpoints stay (fast disk). Datasets come from the fleet folder so101-jepa-data, if it is
# there. outputs/ is copied back to the fleet folder, for `fleet pull`.
SRC=/mnt/c/fleet/so101-jepa
DATA=/mnt/c/fleet/so101-jepa-data
DST=$HOME/so101-jepa
set -e
missing=""
for p in rsync libegl1 libgl1 libosmesa6; do
  dpkg -s $p > /dev/null 2>&1 || missing="$missing $p"
done
if [ -n "$missing" ]; then
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing
fi
mkdir -p "$DST/data" "$DST/outputs"
rsync -a --delete --exclude /data/ --exclude /.venv/ --exclude /outputs/ "$SRC/" "$DST/"
if [ -d "$DATA" ]; then
  rsync -a --exclude "*.fleet-part" "$DATA/" "$DST/data/"
fi
cd "$DST"
set +e
copy_back() {
  mkdir -p "$SRC/outputs"
  rsync -a outputs/ "$SRC/outputs/"
}
# timeout(1) sends TERM to the whole process group: copy the outputs (partial results) back first.
trap 'copy_back; echo "wsl_job: stopped, outputs copied back" >&2; exit 143' TERM INT
"$@"
status=$?
copy_back
exit $status
