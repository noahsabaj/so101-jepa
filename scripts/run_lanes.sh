#!/bin/sh
# Run lanes of jobs on one computer with several GPUs (NVIDIA or AMD, GPU.md), from the project root:
#   sh scripts/run_lanes.sh LANES_FILE
# Each line of LANES_FILE is one lane, "GPU | COMMAND"; the lanes run at the same time, each with
# CUDA_VISIBLE_DEVICES=GPU (ROCm reads it too). Join a lane's steps with && so a failed step ends
# its lane. A step can wait for another lane's output: sh scripts/wait_for FILE LANE (it fails if
# lane LANE ends without FILE; a FILE older than this run, outputs/lanes/start, does not count). Blank lines and lines from # on are skipped.
# Logs: outputs/lanes/N.log; outputs/lanes/N.end holds the lane's exit status. Exit status 1 if any
# lane failed. TERM, INT or HUP to this script is forwarded to every lane and all
# its descendants (steps, training, self-play), and the script waits until they have ended.
mkdir -p outputs/lanes
rm -f outputs/lanes/*.end
touch outputs/lanes/start
n=0 pids=""
killtree() {  # TERM to a process and all its descendants (no job control in a script, so no groups)
  kids=$(pgrep -P "$1" 2> /dev/null)  # listed first: a killed parent no longer has children
  kill -TERM "$1" 2> /dev/null
  for c in $kids; do killtree "$c"; done
}
cancel() {
  trap '' TERM INT HUP
  for p in $pids; do killtree "$p"; done
  for p in $pids; do wait "$p" 2> /dev/null; done
  echo "run_lanes: cancelled, lanes ended" >&2
  exit 1
}
trap cancel TERM INT HUP
while IFS= read -r line; do
  line=${line%%#*}
  [ -n "$(echo "$line" | tr -d ' ')" ] || continue
  gpu=$(echo "${line%%|*}" | tr -d ' ') cmd=${line#*|}
  (
    echo "lane $n on GPU $gpu, $(date '+%F %T'): $cmd"
    CUDA_VISIBLE_DEVICES=$gpu sh -c "$cmd"
    status=$?
    echo "lane $n, $(date '+%F %T'): exit $status"
    echo $status > outputs/lanes/$n.end
    exit $status
  ) > outputs/lanes/$n.log 2>&1 < /dev/null &
  pids="$pids $!" n=$((n + 1))
done < "$1"
fail=0
for p in $pids; do wait "$p" || fail=1; done
exit $fail
