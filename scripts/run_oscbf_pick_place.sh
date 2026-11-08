#!/usr/bin/env bash
# Keep preparation and the complete filtered task in one invocation.
# No arguments previews both plans; --execute performs the recorded task.
# OSCBF_PYTHON reuses an existing interpreter when the source checkout moves.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
oscbf_python="${OSCBF_PYTHON:-$project_root/envs/oscbf/bin/python}"
exec 9>/tmp/oscbf-piper-004F00374548570120353731.lock
flock -n 9

env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$oscbf_python" -I scripts/run_pick_place.py \
  results/pick_place_plan_20261104_224921_126900/task.json \
  --method baseline --speed-scale 1 --descent-speed-scale 1 \
  --stop-after pregrasp --record-diagnostics "$@"

exec env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
  "$oscbf_python" -I scripts/run_pick_place.py \
  results/pick_place_plan_20261107_193629_186524/task.json \
  --method oscbf --speed-scale 1 --descent-speed-scale 1 \
  --record-diagnostics "$@"
