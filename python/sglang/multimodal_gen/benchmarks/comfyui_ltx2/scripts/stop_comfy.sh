#!/bin/bash
# Stop the LTX ComfyUI instance started by start_comfy.sh / mg_session.sh and all its descendants.
PID=$(cat /scratch/data/sgld_comfy/results/ltx25/logs/comfyui.pid 2>/dev/null)
[ -z "$PID" ] && exit 0
desc() { for c in $(pgrep -P $1); do echo $c; desc $c; done; }
TREE=$(desc $PID)
kill $PID 2>/dev/null
for i in $(seq 1 20); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -9 $PID 2>/dev/null
for c in $TREE; do kill -9 $c 2>/dev/null; done
true
