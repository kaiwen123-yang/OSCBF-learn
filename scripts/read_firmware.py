import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from pyAgxArm import (
    AgxArmFactory, ArmModel, PiperFW,
    create_agx_arm_config, resolve_firmware_profile,
)

CHANNEL = "can0"
SERIAL = "004F00374548570120353731"
FIELDS = ("rx_packets", "rx_dropped", "rx_errors",
          "tx_packets", "tx_errors")

def read_stats():
    root = Path("/sys/class/net") / CHANNEL / "statistics"
    return {name: int((root / name).read_text()) for name in FIELDS}

properties = subprocess.check_output(
    ["udevadm", "info", "--query=property",
     f"--path=/sys/class/net/{CHANNEL}"],
    text=True,
)
if f"ID_SERIAL_SHORT={SERIAL}" not in properties.splitlines():
    raise RuntimeError("USB-CAN 身份不匹配，停止查询。")

config = create_agx_arm_config(
    robot=ArmModel.PIPER_H,
    firmeware_version=PiperFW.DEFAULT,
    interface="socketcan",
    channel=CHANNEL,
    bitrate=1000000,
    auto_connect=False,
    enable_check_can=True,
)
robot = AgxArmFactory.create_arm(config)

try:
    robot.connect()
    time.sleep(0.5)

    before = read_stats()
    started = time.monotonic()
    firmware = robot.get_firmware(timeout=2.0)
    print("原始固件信息:", firmware, flush=True)

    time.sleep(5.0)
    after = read_stats()
    report = {
        "time": datetime.now().astimezone().isoformat(),
        "robot": "piper_h",
        "channel": CHANNEL,
        "firmware": firmware,
        "sdk_profile": None,
        "observation_seconds": round(time.monotonic() - started, 3),
        "interface_counter_delta": {
            name: after[name] - before[name] for name in FIELDS
        },
        "sdk_comm_error": str(robot.get_comm_error())
                          if robot.has_comm_error() else None,
    }
    if firmware is not None:
        try:
            report["sdk_profile"] = resolve_firmware_profile(
                ArmModel.PIPER_H, firmware["software_version"].strip()
            )
        except (ValueError, TypeError, KeyError) as exc:
            report["profile_error"] = str(exc)
finally:
    robot.disconnect()

output = Path("logs") / (
    "firmware_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
)
output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
print("已保存:", output.resolve())

if report["sdk_profile"] is None or report["sdk_comm_error"]:
    raise SystemExit("固件查询或通信检查未通过，请回传输出。")
