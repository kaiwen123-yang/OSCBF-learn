#!/usr/bin/env bash
# Execute in a child shell using the system-Python ROS venv, preserving the caller.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 ]]; then
  echo 'Usage: bash scripts/ros_exec.sh command [args...]' >&2
  exit 2
fi
exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
  -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin PYTHONNOUSERSITE=1 LC_NUMERIC=C \
  bash --noprofile --norc -c '
    set -eo pipefail
    project_root="$1"; shift
    source /opt/ros/humble/setup.bash
    source "$project_root/envs/ros-humble/bin/activate"
    source "$project_root/ros_ws/install/local_setup.bash"
    export ROS_LOG_DIR="$project_root/logs/ros"
    mkdir -p "$ROS_LOG_DIR"
    exec "$@"
  ' piper_ros "$project_root" "$@"
