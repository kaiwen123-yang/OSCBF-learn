import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = "can0"
JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)
FIELDS = ("rx_packets", "rx_dropped", "rx_errors", "tx_packets", "tx_errors")

def stats():
    base = Path("/sys/class/net") / CHANNEL / "statistics"
    return {key: int((base / key).read_text()) for key in FIELDS}

properties = subprocess.check_output(
    ["udevadm", "info", "--query=property",
     f"--path=/sys/class/net/{CHANNEL}"], text=True,
)
if "ID_SERIAL_SHORT=004F00374548570120353731" not in properties.splitlines():
    raise RuntimeError("USB-CAN 身份不匹配。")

config = create_agx_arm_config(
    robot=ArmModel.PIPER_H,
    firmeware_version=PiperFW.V189,
    interface="socketcan", channel=CHANNEL, bitrate=1000000,
    auto_connect=False, enable_check_can=True, timeout=0.1,
)
robot = AgxArmFactory.create_arm(config)
rows, pending = [], {}
received_at = {}
counts = {key: 0 for key in JOINT_IDS}
rejected = 0
report = {"robot": "piper_h", "sdk_profile": "v189", "error": None}

try:
    robot.connect(start_read_thread=False)
    comm = robot.get_context().get_comm()
    before = stats()
    started = time.monotonic()

    while time.monotonic() - started < 5.0:
        frame = comm.recv()  # SDK 在返回前完成报文解码
        if frame is None or frame.is_remote_frame or frame.is_error_frame or not frame.is_rx:
            continue
        frame_id = frame.arbitration_id
        received = time.monotonic()
        if len(frame.data) == 8:
            received_at[frame_id] = received
        if frame_id not in counts or len(frame.data) != 8:
            continue
        counts[frame_id] += 1
        pending[frame_id] = (frame.timestamp, received)
        if len(pending) != 3:
            continue

        first, last = min(v[0] for v in pending.values()), max(v[0] for v in pending.values())
        first_mono, last_mono = min(v[1] for v in pending.values()), max(v[1] for v in pending.values())
        pending.clear()
        age = time.monotonic() - first_mono
        if last_mono - first_mono > 0.020 or not 0 <= age <= 0.100:
            rejected += 1
            continue

        feedback = robot.get_joint_angles()
        q = list(feedback.msg) if feedback is not None else []
        if len(q) != 6 or not all(math.isfinite(x) for x in q):
            rejected += 1
            continue
        rows.append([time.monotonic() - started, first, last, *q, first_mono, last_mono])

    after = stats()
    report["interface_counter_delta"] = {
        key: after[key] - before[key] for key in FIELDS
    }
    report["joint_packet_counts"] = {
        f"0x{key:03X}": value for key, value in counts.items()
    }
    report["complete_samples"] = len(rows)
    report["rejected_groups"] = rejected
    if len(rows) < 2:
        raise RuntimeError("完整且新鲜的六关节反馈不足。")

    now = time.monotonic()
    drivers = [robot.get_driver_states(j) for j in range(1, 7)]
    status = robot.get_arm_status()
    messages = drivers + [status]
    if any(msg is None for msg in messages):
        raise RuntimeError("驱动器或机械臂状态反馈不完整。")
    if any(not 0 <= now-received_at.get(i,-math.inf) <= 0.2 for i in (*range(0x261,0x267),0x2A1)):
        raise RuntimeError("驱动器或机械臂状态反馈过期。")
    if not 0 <= now - rows[-1][9] <= 0.1:
        raise RuntimeError("末次六关节反馈过期。")
    if robot.has_comm_error():
        raise RuntimeError(str(robot.get_comm_error()))

    gaps = [b[10] - a[10] for a, b in zip(rows, rows[1:])]
    report.update({
        "joint_rad_last": rows[-1][3:9],
        "joint_deg_last": [round(math.degrees(x), 4) for x in rows[-1][3:9]],
        "feedback_timing_clock": "monotonic_receive",
        "complete_feedback_hz": round(
            (len(rows) - 1) / (rows[-1][10] - rows[0][10]), 2),
        "max_feedback_gap_ms": round(max(gaps) * 1000, 3),
        "joint_enabled": [msg.msg.foc_status.driver_enable_status
                          for msg in drivers],
        "driver_faults": {
            f"J{j}": [key for key, value in msg.msg.foc_status.to_dict().items()
                     if key != "driver_enable_status" and value]
            for j, msg in enumerate(drivers, 1)
        },
        "arm_status": status.msg.to_dict(),
    })
except (Exception, KeyboardInterrupt) as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
finally:
    robot.disconnect()

stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
csv_path = ROOT / "logs" / f"joint_state_{stamp}.csv"
json_path = ROOT / "logs" / f"joint_state_{stamp}.json"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["elapsed_s", "rx_first_unix_s", "rx_last_unix_s"] +
                    [f"q{j}_rad" for j in range(1, 7)] +
                    ["rx_first_monotonic_s", "rx_last_monotonic_s"])
    writer.writerows(rows)
json_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("CSV:", csv_path)
print("JSON:", json_path)
if report["error"]:
    raise SystemExit(1)
