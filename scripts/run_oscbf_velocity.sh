#!/usr/bin/env bash
# Official Panda velocity example; never connects to Piper H.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root/src/oscbf"
test "$(git rev-parse HEAD)" = 6082329f8e1d61ee8e875a63a5cd80a14d513e12
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  "$project_root/envs/oscbf/bin/python" -I -m oscbf.examples.dynamic_motion \
  --control_method velocity
