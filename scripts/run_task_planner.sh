#!/usr/bin/env bash
# Offline MoveIt service backend only: no driver, no CAN, no control gate.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_config="${1:-$project_root/config/pick_place.json}"
tcp_offset=$(/usr/bin/python3 - "$task_config" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1])); values=cfg['tcp_offset_m']+cfg.get('tcp_rpy_rad',[0.,0.,0.])
print(json.dumps(values,separators=(',',':')))
PY
)
export ROS_DOMAIN_ID=78 ROS_LOCALHOST_ONLY=1
exec bash "$project_root/scripts/ros_exec.sh" ros2 launch agx_arm_moveit demo.launch.py \
  arm_type:=piper_h effector_type:=agx_gripper namespace:=piper_task \
  tcp_offset:="$tcp_offset" \
  follow:=false auto_control_gate:=false db:=false use_rviz:=false
