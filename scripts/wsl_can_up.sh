#!/usr/bin/env bash
# Configure only the serial-matched Piper CAN interface. Sends no robot commands.
set -euo pipefail
if [[ $# != 0 ]]; then echo 'Usage: sudo bash scripts/wsl_can_up.sh' >&2; exit 2; fi
if [[ $EUID != 0 ]]; then echo 'Run this interface setup as root.' >&2; exit 1; fi
case "$(uname -r)" in
  *-microsoft-standard-WSL2) ;;
  *) echo 'This helper is for WSL2 only.' >&2; exit 1 ;;
esac
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# WSL mounts its modules through an overlay. Restore only this project's modules
# if a WSL restart discarded the overlay, and never load them into a new kernel.
if ! modinfo gs_usb >/dev/null 2>&1; then
  release=$(uname -r)
  driver="$project_root/.wsl/driver"
  for module in can-dev gs_usb; do
    test -f "$driver/$module.ko" || { echo 'Project CAN modules are missing.' >&2; exit 1; }
    [[ "$(modinfo -F vermagic "$driver/$module.ko")" == "$release "* ]] || {
      echo 'WSL kernel changed; rebuild the CAN modules for this kernel first.' >&2; exit 1;
    }
  done
  if [[ -d /sys/module/can_dev && ! -d /sys/module/gs_usb ]]; then
    echo 'Another CAN device is already using can_dev; refusing to replace it.' >&2; exit 1
  fi
  for module in can-dev gs_usb; do
    install -D -m 0644 "$driver/$module.ko" "/lib/modules/$release/updates/oscbf/$module.ko"
  done
  depmod -a "$release"
fi
modprobe can_raw
modprobe gs_usb
# usbipd attach returns before USB enumeration has necessarily completed.
for ((attempt=0; attempt<50; attempt++)); do
  [[ -d /sys/class/net/can0 ]] && break
  sleep 0.2
done
udevadm settle --timeout=10
test -d /sys/class/net/can0 || { echo 'can0 is absent; attach the Piper USB-CAN from Windows first.' >&2; exit 1; }
properties=$(udevadm info --query=property --path=/sys/class/net/can0)
if ! grep -qx 'ID_SERIAL_SHORT=004F00374548570120353731' <<< "$properties"; then
  echo 'Piper USB-CAN serial mismatch; interface left untouched.' >&2
  exit 1
fi
# Do not interrupt an interface that is already up or change an active bitrate.
state=$(ip -details -json link show dev can0)
if /usr/bin/python3 -c 'import json,sys; sys.exit("UP" not in json.load(sys.stdin)[0]["flags"])' <<< "$state"; then
  /usr/bin/python3 -c '
import json, sys
info = json.load(sys.stdin)[0]["linkinfo"]["info_data"]
assert info.get("bittiming", {}).get("bitrate") == 1000000, "Active can0 bitrate is not 1 Mbps; left untouched"
assert "LISTEN-ONLY" not in info.get("ctrlmode", []), "Active can0 is listen-only; left untouched"
assert info.get("state") != "BUS-OFF", "can0 is bus-off; investigate before restarting it"
' <<< "$state"
else
  ip link set dev can0 type can bitrate 1000000
  ip link set dev can0 up
fi
ip -details -statistics link show dev can0
echo 'PIPER_CAN_READY: 1 Mbps, matching serial; no enable, reset or motion command sent.'
