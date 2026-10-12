#!/usr/bin/env bash
# Print the candidate data disk with the most free space (use it for WORKSPACE in kit.env).
# Needs ~850 GB on 4090/5090 and ~1.1 TB on h200/b200 (track A + B models, caches, results).
best=""; best_free=0
for d in /personal /scratch /data /mnt "$HOME"; do
  [ -d "$d" ] && [ -w "$d" ] || continue
  free=$(df -Pk "$d" | awk 'NR==2 {print $4}')
  printf "%-10s free %6d GB\n" "$d" $((free / 1024 / 1024))
  if [ "$free" -gt "$best_free" ]; then best=$d; best_free=$free; fi
done
echo "PICK $best ($((best_free / 1024 / 1024)) GB free) -> WORKSPACE=$best/sgld_bench"
