#!/usr/bin/env bash
# Keep preparation and the complete filtered task in one WSL invocation.
# No arguments previews both plans; --execute performs the recorded task.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec 9>/tmp/oscbf-piper-004F00374548570120353731.lock
flock -n 9

env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$project_root/envs/oscbf/bin/python" -I scripts/run_pick_place.py \
  results/pick_place_plan_20261104_224921_126900/task.json \
  --method baseline --speed-scale 1 --descent-speed-scale 1 \
  --stop-after pregrasp --record-diagnostics "$@"

exec env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$project_root/envs/oscbf/bin/python" -I scripts/run_pick_place.py \
  results/pick_place_plan_20261107_193629_186524/task.json \
  --method oscbf --speed-scale 1 --descent-speed-scale 1 \
  --record-diagnostics "$@"
