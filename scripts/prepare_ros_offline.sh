#!/usr/bin/env bash
# User-run only. Downloads/builds software; never starts a robot driver.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != --clean-child ]]; then
  exec env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u VIRTUAL_ENV \
    -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin PYTHONNOUSERSITE=1 LC_NUMERIC=C \
    bash "$0" --clean-child "$@"
fi
shift
allow_apt=false
if [[ "${1:-}" == --install-system-deps ]]; then allow_apt=true; shift; fi
if [[ $# -ne 0 ]]; then echo 'Unknown argument' >&2; exit 2; fi
cd "$project_root"
mkdir -p logs ros_ws/src models
run_log="$project_root/logs/ros_prepare_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$run_log") 2>&1
trap 'rc=$?; echo "FAILED at line $LINENO (exit $rc); log: $run_log"; exit "$rc"' ERR
ros_commit=286633e0d2edb038ee3a1047d9feb7f614a1c076
urdf_commit=680fa925cd00db7e2628563d2edc5540963045de
sdk_commit=d90e234ba8e408cfa964b38aa9120b3131fc9f85
repo="$project_root/ros_ws/src/agx_arm_ros"
venv="$project_root/envs/ros-humble"
sdk_python="$project_root/envs/piper-sdk/bin/python"
test "$(git -C "$project_root/src/pyAgxArm" rev-parse HEAD)" = "$sdk_commit"
if [[ ! -e "$repo/.git" ]]; then
  git clone --branch ros2 --no-checkout https://github.com/agilexrobotics/agx_arm_ros.git "$repo"
  git -C "$repo" checkout -B ros2 "$ros_commit"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$ros_commit" || {
  echo 'ROS checkout differs from audited commit; not overwriting it.'; exit 1;
}
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
git -C "$repo" submodule update --init --recursive
test "$(git -C "$repo/src/agx_arm_description/agx_arm_urdf" rev-parse HEAD)" = "$urdf_commit"
{
  git -C "$repo" log -1 --format='%H%n%cI%n%s'
  git -C "$repo" submodule status --recursive
} > "$project_root/logs/ros_source_versions.txt"

packages=(build-essential cmake python3-venv python3-numpy python3-scipy python3-colcon-common-extensions
  ros-humble-rosidl-default-generators ros-humble-xacro
  ros-humble-joint-state-publisher-gui ros-humble-moveit
  ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-topic-tools)
missing=()
for package in "${packages[@]}"; do
  if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != 'install ok installed' ]]; then
    missing+=("$package")
  fi
done
printf '%s\n' "${missing[@]}" > "$project_root/logs/ros_missing_apt.txt"
if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'Missing system/ROS packages: %s\n' "${missing[*]}"
  if [[ "$allow_apt" != true ]]; then
    echo 'Rerun with --install-system-deps to install this listed set.'; exit 2
  fi
  sudo apt-get update
  sudo apt-get install --no-install-recommends "${missing[@]}"
fi

if [[ ! -x "$venv/bin/python" ]]; then
  /usr/bin/python3 -m venv --system-site-packages --without-pip "$venv"
fi
"$venv/bin/python" -I -c 'import sys; assert sys.base_prefix == "/usr"; assert sys.prefix != sys.base_prefix; print("ROS Python:", sys.executable, "base:", sys.base_prefix)'
"$sdk_python" -I -m pip --python "$venv/bin/python" install \
  --no-deps -r "$project_root/requirements/ros-overlay.txt" "$project_root/src/pyAgxArm"
source /opt/ros/humble/setup.bash
source "$venv/bin/activate"
export COLCON_PYTHON_EXECUTABLE="$venv/bin/python"
"$venv/bin/python" -c 'import rclpy, pyAgxArm; from scipy.spatial.transform import Rotation; print("ROS/SDK/scipy imports OK")'
cd "$project_root/ros_ws"
colcon_entry="$(command -v colcon)"
test -f "$colcon_entry"
"$venv/bin/python" "$colcon_entry" build --symlink-install --parallel-workers 2 \
  --event-handlers console_direct+ \
  --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE="$venv/bin/python" \
    -DPYTHON_EXECUTABLE="$venv/bin/python"
source "$project_root/ros_ws/install/local_setup.bash"
"$venv/bin/python" -c 'from pathlib import Path; from agx_arm_msgs.msg import MoveMITMsg; from ament_index_python.packages import get_package_prefix; entry = Path(get_package_prefix("agx_arm_ctrl"))/"lib/agx_arm_ctrl/agx_arm_ctrl_single"; assert entry.is_file(), entry; print("Built message import and driver entrypoint OK; driver not started")'
"$venv/bin/python" "$project_root/scripts/export_piper_model.py"
"$sdk_python" -I -m pip --python "$venv/bin/python" freeze > "$project_root/logs/ros_python_freeze.txt"
dpkg-query -W -f='${Package}\t${Version}\n' "${packages[@]}" > "$project_root/logs/ros_apt_versions.txt"
echo "ROS_OFFLINE_BUILD_OK"
echo "LOG: $run_log"
