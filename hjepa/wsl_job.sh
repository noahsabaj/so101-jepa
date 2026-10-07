#!/bin/sh
# Run a command for this project in WSL on kat-pc (Linux, CUDA), from a fleet job there:
#   fleet run --on kat-pc --gpu-gb 7 -- wsl.exe -d DISTRO -- sh hjepa/wsl_job.sh CMD ARGS...
# The fleet copy (/mnt/c/fleet/so101-jepa) is synced to ext4 (~/so101-jepa), where the venv, data
# and checkpoints stay (fast disk). Datasets come from the fleet folder so101-jepa-data, if it is
# there. outputs/ is copied back to the fleet folder, for `fleet pull`.
SRC=$(cd "$(dirname "$0")/.." && pwd)  # the fleet copy, e.g. /mnt/c/fleet/so101-jepa
DATA=/mnt/c/fleet/so101-jepa-data
DST=$HOME/$(basename "$SRC")
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
# The command runs in its own session, so this script can end all of it: on TERM (timeout(1)), or
# when the fleet job is stopped. A stop kills wsl.exe on Windows but not the Linux processes; then
# the heartbeat cannot write to stdout, and the script ends the command. Outputs are copied back.
setsid "$@" &
pid=$!
stop() {
  kill -TERM -$pid 2>/dev/null
  sleep 10
  kill -KILL -$pid 2>/dev/null
  copy_back
  echo "wsl_job: $1; outputs copied back" >&2
  exit 143
}
trap 'stop "stopped by a signal"' TERM INT HUP
trap '' PIPE
while kill -0 $pid 2>/dev/null; do
  sleep 15
  date '+wsl_job: running %H:%M:%S' 2>/dev/null || stop "stdout is closed (the fleet job was stopped)"
done
wait $pid
status=$?
copy_back
exit $status
