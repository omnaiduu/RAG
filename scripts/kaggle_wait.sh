#!/usr/bin/env bash
# Usage: scripts/kaggle_wait.sh <kernel-slug> <out-dir>   -- polls until done, then downloads outputs.
set -u
slug=$1; out=$2; mkdir -p "$out"
for i in $(seq 1 720); do
  s=$(timeout 60 kaggle kernels status "$slug" 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "$s"; break;;
  esac
  sleep 30
done
timeout 600 kaggle kernels output "$slug" -p "$out" >/dev/null 2>&1 || timeout 600 kaggle kernels output "$slug" -p "$out" 2>&1 | tail -2
ls "$out"
