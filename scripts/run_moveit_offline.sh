#!/usr/bin/env bash
# Isolated mock MoveIt only: does not launch agx_arm_ctrl or open SocketCAN.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$project_root/ros_ws/src/agx_arm_ros"
test "$(git -C "$repo" rev-parse HEAD)" = 286633e0d2edb038ee3a1047d9feb7f614a1c076
git -C "$repo" diff --exit-code -- src/agx_arm_moveit
git -C "$repo" diff --cached --exit-code -- src/agx_arm_moveit
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
mkdir -p "$project_root/logs"
echo 'OFFLINE MOCK: Piper H + gripper; ROS domain 77. This is not physical feedback.'
bash "$project_root/scripts/ros_exec.sh" ros2 launch agx_arm_moveit demo.launch.py \
  arm_type:=piper_h effector_type:=agx_gripper namespace:=oscbf_offline \
  follow:=false auto_control_gate:=false db:=false use_rviz:=true \
  2>&1 | tee "$project_root/logs/moveit_offline_$(date +%Y%m%d_%H%M%S).log"
