#!/bin/sh
# Progress of the runs in WSL on kat-pc: the last progress line of each training log, recent
# errors, and the GPU. Usage (from a fleet job on kat-pc):
#   wsl.exe -d Ubuntu-24.04 -- sh /mnt/c/fleet/so101-jepa/hjepa/peek.sh
cd "$HOME/so101-jepa/outputs" || exit 1
for f in train_*.log; do
  [ -f "$f" ] || continue
  echo "== $f ($(date -r "$f" +%H:%M))"
  tail -c 20000 "$f" | tr '\r' '\n' | grep -E "Epoch [0-9]+/" | tail -1 | cut -c1-200
  tail -c 20000 "$f" | grep -E "Error|Traceback|out of memory" | tail -2
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
