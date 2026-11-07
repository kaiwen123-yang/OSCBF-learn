#!/usr/bin/env bash
# New Ubuntu 22.04/Python 3.10 checkout only. Installs software, never opens CAN.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_ros=false
if [[ "${1:-}" == --with-ros ]]; then build_ros=true; shift; fi
if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/bootstrap_new_device.sh [--with-ros]' >&2; exit 2; fi
cd "$project_root"
python_bin="${PIPER_PYTHON:-python3.10}"
"$python_bin" -I -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
git submodule update --init --recursive
mkdir -p envs logs
if [[ ! -x envs/piper-sdk/bin/python ]]; then
  "$python_bin" -m venv envs/piper-sdk
fi
envs/piper-sdk/bin/python -I -m pip install -r requirements/piper-sdk.txt
envs/piper-sdk/bin/python -I -m pip install --no-deps -e ./src/pyAgxArm
envs/piper-sdk/bin/python -I -m pip check
bash scripts/prepare_oscbf.sh
if [[ "$build_ros" == true ]]; then
  bash scripts/prepare_ros_offline.sh --install-system-deps
fi
echo 'NEW_DEVICE_SOFTWARE_READY: no CAN connection, enable, home or motion was performed.'
