"""Read Piper H status, or clear software estop without enabling or moving.

--reset-only --execute sends one reset. Reset removes motor power: the movable
arm must be physically supported. There is no automatic hold or home step.
"""
import argparse
import csv
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_oscbf.device import PiperDevice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--reset-only", action="store_true",
                        help="Clear software estop only; removes motor power, no enable or motion")
    action.add_argument("--clear-joint-error", type=int, choices=range(1,7),
                        help="Clear the specified driver error only, without reset/enable/motion")
    args = parser.parse_args()
    if args.execute and not (args.reset_only or args.clear_joint_error):
        parser.error("Choose --reset-only or --clear-joint-error N. Automatic enable/motion recovery is removed.")
    device = PiperDevice()
    report = {"execute": args.execute, "status": "read_only", "error": None,
              "reset_sent": False, "enable_sent": False, "motion_sent": False}
    snapshots = []

    def snapshot():
        status = device.robot.get_arm_status()
        joints = device.robot.get_joint_angles()
        if device.robot.has_comm_error():
            raise RuntimeError(str(device.robot.get_comm_error()))
        if (status is None or joints is None or
                max(device.message_age(i) for i in (0x2A1,0x2A5,0x2A6,0x2A7)) > .2):
            raise RuntimeError("Fresh arm/joint feedback unavailable")
        motors = [device.robot.get_driver_states(j) for j in range(1, 7)]
        result = {"time_s": time.time(), "monotonic_s": time.monotonic(),
                  "status_timestamp_s": status.timestamp,
                  "status_received_monotonic_s": device.received_at.get(0x2A1),
                  "q_rad": list(joints.msg), "status": status.msg.to_dict(),
                  "enabled": [bool(m.msg.foc_status.driver_enable_status) if m else None for m in motors],
                  "driver_status_codes": [m.msg.foc_status_code if m else None for m in motors],
                  "motor_status_timestamps_s": [m.timestamp if m else None for m in motors],
                  "motor_received_monotonic_s": [device.received_at.get(0x260+j) for j in range(1,7)]}
        snapshots.append(result)
        return result

    def print_state(label, state):
        print(label, "deg:", [round(math.degrees(v), 3) for v in state["q_rad"]],
              "arm_status:", int(state["status"]["arm_status"]),
              "enabled:", state["enabled"], "driver_codes:",state["driver_status_codes"],flush=True)

    def terminate(*_):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        device.connect()
        initial = report["initial"] = snapshot()
        print_state("当前", initial)
        if args.execute and args.clear_joint_error:
            j=args.clear_joint_error
            print(f"仅清除J{j}驱动错误码，不reset、不使能、不发送运动目标。",flush=True)
            device.phase="clear_joint_error_only"
            report["joint_error_clear_requested"]=j
            report["joint_error_clear_ack"]=device.robot.clear_joint_error(j)
            for _ in range(20):
                time.sleep(.05)
                state=snapshot()
            code=state["driver_status_codes"][j-1]
            if code is None or code & 0xBF:
                raise RuntimeError(f"J{j} driver error remains: {code}; no enable or motion sent")
            report["status"]="joint_error_cleared_no_motion"
            print("JOINT_ERROR_CLEARED: 新反馈故障位已清除，没有使能或运动。",flush=True)
        elif args.execute:
            arm_status = int(initial["status"]["arm_status"])
            if arm_status == 1:
                print("仅清除软件急停：reset会立即失电。不会使能、回零或操作夹爪。", flush=True)
                device.phase = "reset_only"
                stamp = started = time.monotonic()
                device.robot.reset()
                report["reset_sent"] = True
                confirmed = False
                while time.monotonic()-started < 5:
                    time.sleep(.05)
                    state = snapshot()
                    confirmed = (state["status_received_monotonic_s"] > stamp and
                                 int(state["status"]["arm_status"]) == 0 and
                                 all(t is not None and t > stamp for t in state["motor_received_monotonic_s"]) and
                                 all(v is False for v in state["enabled"]))
                    if confirmed and time.monotonic()-started >= 3:
                        break
                if not confirmed:
                    raise RuntimeError("Reset not confirmed; no enable or motion sent")
                report["status"] = "reset_only_completed"
                print("RESET_ONLY_FINISHED: 已清除急停，六轴失能；没有发送运动目标。", flush=True)
            elif arm_status == 0:
                report["status"] = "already_normal_no_command"
                print("软件急停已清除，本次没有发送命令。", flush=True)
            else:
                raise RuntimeError(f"Controller status {arm_status}; no reset sent")
        else:
            print("READ_ONLY: 未发送复位、使能、运动或夹爪指令。", flush=True)
        report["final"] = snapshot()
        print_state("结束", report["final"])
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        # This entry never starts control. Do not add another estop or enable.
        print(report["error"], flush=True)
    finally:
        device.disconnect()
        stem = ROOT/"logs"/("task_recovery_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        report["snapshots"] = snapshots
        stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        if device.feedback:
            with stem.with_suffix(".csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=device.feedback[0].keys())
                writer.writeheader()
                writer.writerows(device.feedback)
        print("RESULT:", stem.with_suffix(".json"), flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
