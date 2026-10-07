# The GPU vendor of this computer and the matching torch build (GPU.md). Scripts source it:
#   . "$(dirname "$0")/../scripts/gpu.sh"
# It sets and exports
#   GPU        cuda (NVIDIA, the default) or rocm (AMD: /dev/kfd and no nvidia-smi); SO101_GPU overrides
#   UV_GROUPS  the uv options for that torch build: none for cuda, "--no-group cuda --group rocm" for rocm
#   GL         the default MuJoCo renderer: egl on NVIDIA; osmesa (CPU) on AMD, as Instinct GPUs have
#              no graphics (a Radeon card can set MUJOCO_GL=egl)
# and defines
#   uvr ARGS...  uv run with this GPU's torch build
#   gpu_sample   "memory used (MiB),use (%)" of GPU 0, all processes ("?" if unknown)
if [ -n "${SO101_GPU:-}" ]; then
  GPU=$SO101_GPU
elif command -v nvidia-smi > /dev/null 2>&1; then
  GPU=cuda
elif [ -e /dev/kfd ]; then
  GPU=rocm
else
  GPU=cuda
fi
case $GPU in
  rocm) UV_GROUPS="--no-group cuda --group rocm" GL=osmesa ;;
  cuda) UV_GROUPS="" GL=egl ;;
  *) echo "gpu.sh: SO101_GPU must be cuda or rocm, not $GPU" >&2; exit 2 ;;
esac
export GPU UV_GROUPS GL

uvr() {
  uv run $UV_GROUPS "$@"
}

gpu_sample() {
  if [ "$GPU" = cuda ]; then
    nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' '
  else  # rocm-smi CSV: find the columns by name (their order changes between ROCm versions)
    rocm-smi -d 0 --showmeminfo vram --showuse --csv 2> /dev/null | awk -F, '
      NR == 1 { for (i = 1; i <= NF; i++) { if ($i ~ /Used Memory/) m = i; if ($i ~ /GPU use/) u = i } next }
      NR == 2 { printf "%s,%s\n", m ? int($m / 1048576) : "?", u ? $u : "?" }'
  fi
}
