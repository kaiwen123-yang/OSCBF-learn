"""Read controller limits; never enable, move, reset, or configure the arm."""
import json
import math
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_COMMIT = "d90e234ba8e408cfa964b38aa9120b3131fc9f85"


def in_bounds(value, lower, upper):
    return (all(isinstance(x, (int, float)) and math.isfinite(x)
                for x in (value, lower, upper))
            and lower < upper and lower <= value <= upper)


def main():
    from pyAgxArm import (AgxArmFactory, ArmModel, PiperFW,
                         create_agx_arm_config, resolve_firmware_profile)

    props = subprocess.check_output(
        ["udevadm", "info", "--query=property", "--path=/sys/class/net/can0"], text=True)
    if "ID_SERIAL_SHORT=004F00374548570120353731" not in props.splitlines():
        raise RuntimeError("USB-CAN identity mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "src/pyAgxArm"), "rev-parse", "HEAD"], text=True).strip()
    if commit != SDK_COMMIT:
        raise RuntimeError("SDK source commit changed; re-audit before querying")
    config = create_agx_arm_config(
        robot=ArmModel.PIPER_H, firmeware_version=PiperFW.V189,
        interface="socketcan", channel="can0", bitrate=1000000,
        auto_connect=False, enable_check_can=True, timeout=0.1)
    robot = AgxArmFactory.create_arm(config)
    latest, pending = {}, {}
    lock = threading.Lock()

    # Registered AFTER the SDK parser: this callback copies complete parsed data.
    def observe(frame):
        if (frame.arbitration_id not in (0x2A5, 0x2A6, 0x2A7)
                or len(frame.data) != 8 or not frame.is_rx
                or frame.is_remote_frame or frame.is_error_frame):
            return
        pending[frame.arbitration_id] = frame.timestamp
        if len(pending) != 3:
            return
        first, last = min(pending.values()), max(pending.values())
        pending.clear()
        if last - first > 0.02 or not 0 <= time.time() - first <= 0.1:
            return
        msg = robot.get_joint_angles()
        if msg is not None and len(msg.msg) == 6 and all(math.isfinite(x) for x in msg.msg):
            with lock:
                latest.update(q=list(msg.msg), timestamp=first)

    def current_q():
        with lock:
            if not latest or not 0 <= time.time() - latest["timestamp"] <= 0.1:
                raise RuntimeError("Complete fresh joint feedback unavailable")
            return latest["q"][:]

    def counters():
        base = Path("/sys/class/net/can0/statistics")
        return {k: int((base / k).read_text()) for k in
                ("rx_packets", "rx_dropped", "rx_errors", "tx_packets", "tx_errors")}

    report = {"time": datetime.now().astimezone().isoformat(),
              "sdk_commit": commit, "robot": "piper_h", "channel": "can0",
              "queries_only": True, "joints": [], "errors": []}
    robot.get_context().register_parser_packet_fun(observe)
    try:
        robot.connect()
        time.sleep(0.5)
        current_q()
        before = counters()
        report["firmware"] = robot.get_firmware(timeout=2.0)
        if report["firmware"] is None:
            raise RuntimeError("Firmware query timed out")
        profile = resolve_firmware_profile(
            ArmModel.PIPER_H, report["firmware"]["software_version"].strip())
        report["sdk_profile"] = profile
        if profile != "v189":
            raise RuntimeError("Firmware profile differs from previously verified v189")
        print("J  current(deg)  controller[min,max](deg)  SDK[min,max](deg)", flush=True)
        for j in range(1, 7):
            limits = robot.get_joint_angle_vel_limits(j, timeout=2.0)
            accel = robot.get_joint_acc_limits(j, timeout=2.0)
            q = current_q()[j - 1]
            lo, hi = config["joint_limits"][f"joint{j}"]
            row = {"joint": j, "q_rad": q, "q_deg": math.degrees(q),
                   "sdk_bounds_rad": [lo, hi], "inside_sdk_bounds": in_bounds(q, lo, hi),
                   "controller_limits": limits.msg.to_dict() if limits else None,
                   "controller_acceleration": accel.msg.to_dict() if accel else None,
                   "inside_controller_bounds": None}
            if limits is not None:
                low, high = limits.msg.min_angle_limit, limits.msg.max_angle_limit
                row["inside_controller_bounds"] = in_bounds(q, low, high)
                bounds = [round(math.degrees(v), 4) if v is not None else None
                          for v in (low, high)]
            else:
                bounds = None
                report["errors"].append(f"J{j} angle/velocity query timed out")
            if accel is None:
                report["errors"].append(f"J{j} acceleration query timed out")
            report["joints"].append(row)
            print(f"J{j} {math.degrees(q):10.4f}  {bounds}  "
                  f"{[round(math.degrees(v), 4) for v in (lo, hi)]}", flush=True)
        after = counters()
        report["counter_delta"] = {k: after[k] - before[k] for k in before}
        report["q_rad_last"] = current_q()
        if robot.has_comm_error():
            report["errors"].append(str(robot.get_comm_error()))
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        robot.disconnect()
    report["outside_sdk_bounds"] = [r["joint"] for r in report["joints"]
                                    if not r["inside_sdk_bounds"]]
    report["outside_controller_bounds"] = [r["joint"] for r in report["joints"]
                                           if r["inside_controller_bounds"] is False]
    report["motion_performed"] = False
    output = ROOT / "logs" / ("controller_limits_" +
                             datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SAVED:", output)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
