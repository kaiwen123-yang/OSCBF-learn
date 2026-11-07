"""Piper H J6 velocity-interface demonstration, started by the operator.

0 -> +0.02 rad/s (2 s) -> 0 (1 s) -> -0.02 (2 s) -> 0 (1 s).
J1-J5 retain their measured initial position targets. Finish in MOVE_J
at the initial measured configuration, 10% speed. No Flash parameters changed.
"""
import argparse
import csv
import json
import math
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    props = subprocess.check_output(
        ["udevadm", "info", "--query=property", "--path=/sys/class/net/can0"], text=True)
    if "ID_SERIAL_SHORT=004F00374548570120353731" not in props.splitlines():
        raise RuntimeError("Piper H USB-CAN identity mismatch")
    robot = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.PIPER_H, firmeware_version=PiperFW.V189,
        interface="socketcan", channel="can0", bitrate=1000000,
        auto_connect=False, enable_check_can=True, timeout=.05))
    lock, pending, rows, commands = threading.Lock(), {}, [], []
    phase, commanded_vel = "observe", 0.0

    def observe(frame):
        if not frame.is_rx or frame.is_error_frame or frame.is_remote_frame:
            return
        if frame.arbitration_id not in (0x2A5, 0x2A6, 0x2A7) or len(frame.data) != 8:
            return
        pending[frame.arbitration_id] = (time.monotonic(), frame.timestamp)
        if len(pending) == 3:
            oldest, stamp = min(pending.values())
            pending.clear()
            with lock:
                rows.append([oldest, stamp, phase, commanded_vel,
                             *list(robot.get_joint_angles().msg)])

    def state():
        if robot.has_comm_error():
            raise RuntimeError(str(robot.get_comm_error()))
        with lock:
            if not rows:
                raise RuntimeError("No complete six-joint feedback")
            row = rows[-1][:]
        # Communication timeout: do not continue a velocity command on frozen data.
        if time.monotonic()-row[0] > .2:
            raise RuntimeError("Joint feedback interrupted")
        status = robot.get_arm_status()
        if status is None or time.time()-status.timestamp > .2:
            raise RuntimeError("Arm status feedback interrupted")
        if int(status.msg.arm_status) != 0:
            raise RuntimeError(f"Controller stopped/faulted: {status.msg}")
        if not all(robot.get_joints_enable_status_list()):
            raise RuntimeError("Arm is not fully enabled; run the working home entry first")
        return row[4:], status.msg.to_dict()

    robot.get_context().register_parser_packet_fun(observe)
    report = {"robot": "piper_h", "firmware_profile": "v189", "joint": 6,
              "velocity_rad_s": .02, "phases": [], "error": None,
              "note": "Measured joint feedback; no automatic motion-quality pass/fail threshold."}
    control_started = False

    def terminate(*_):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)

    def send_velocity(value):
        nonlocal commanded_vel
        robot.move_cpv_vel(6, value)
        commanded_vel = value
        commands.append([time.monotonic(), phase, value])

    def velocity_phase(name, value, duration):
        nonlocal phase
        phase = name
        q_start, _ = state()
        start = time.monotonic()
        while time.monotonic()-start < duration:
            q, status = state()
            if name != "zero_before" and int(status["mode_feedback"]) != 5:
                raise RuntimeError(f"CPV mode not confirmed: {status}")
            send_velocity(value)
            time.sleep(.01)
        # End every phase with an explicit zero before reporting or changing direction.
        send_velocity(0.0)
        elapsed = time.monotonic()-start
        q_end, status = state()
        delta = [math.degrees(b-a) for a, b in zip(q_start, q_end)]
        summary = {"phase": name, "command_rad_s": value, "duration_s": elapsed,
                   "q_start_rad": q_start, "q_end_rad": q_end,
                   "delta_deg": delta, "mean_j6_rad_s": (q_end[5]-q_start[5])/elapsed,
                   "controller_status": status}
        report["phases"].append(summary)
        print(f"{name}: 指令={value:+.3f} rad/s, J6变化={delta[5]:+.4f} deg, "
              f"平均速度={summary['mean_j6_rad_s']:+.5f} rad/s, "
              f"全部角度={[round(math.degrees(x), 3) for x in q_end]}", flush=True)

    try:
        robot.connect()
        time.sleep(.6)
        q0, status = state()
        report.update(initial_rad=q0, initial_status=status)
        print("初始实测 deg:", [round(math.degrees(x), 3) for x in q0], flush=True)
        if args.execute:
            # CPV position API clamps invalid targets: reject rather than silently alter them.
            for j, q in enumerate(q0, 1):
                lo, hi = robot.get_config()["joint_limits"][f"joint{j}"]
                if not lo <= q <= hi:
                    raise RuntimeError(f"J{j} initial position outside SDK bounds; do not start CPV")
            control_started = True
            robot.set_motion_mode("cpv")
            robot.set_auto_set_motion_mode_enabled(False)
            for j in range(1, 6):
                robot.move_cpv_pos(j, q0[j-1])
            velocity_phase("zero_before", 0.0, 1.0)
            velocity_phase("positive", .02, 2.0)
            velocity_phase("zero_after_positive", 0.0, 1.0)
            velocity_phase("negative", -.02, 2.0)
            velocity_phase("zero_after_negative", 0.0, 1.0)
            phase = "position_return"
            send_velocity(0.0)
            robot.set_speed_percent(10)
            robot.set_motion_mode("j")
            for _ in range(50):
                state()
                robot.move_j(q0)
                time.sleep(.02)
            q_end, status = state()
            report.update(final_rad=q_end, final_status=status, completed_sequence=True)
            print("CPV_SEQUENCE_FINISHED: 已发送零速度并切回初始位置目标，保持使能。", flush=True)
        else:
            print("READ_ONLY: 未发送控制指令；加 --execute 执行。", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if control_started:
            try:
                send_velocity(0.0)
            except Exception as stop_error:
                report["zero_send_error"] = str(stop_error)
            try:
                robot.electronic_emergency_stop()
                print("已发送电子急停，可能下沉；现场确认停止。", flush=True)
            except Exception as stop_error:
                report["estop_send_error"] = str(stop_error)
                print("电子急停未能发送，请立即使用物理急停。", flush=True)
        print("ERROR:", report["error"], flush=True)
    finally:
        robot.disconnect()
        stem = ROOT / "logs" / ("cpv_j6_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        stem.parent.mkdir(exist_ok=True)
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["monotonic_s", "can_timestamp_s", "phase", "command_j6_rad_s"]
                            + [f"q{i}_rad" for i in range(1, 7)])
            writer.writerows(rows)
        with Path(str(stem)+"_commands.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["monotonic_s", "phase", "command_j6_rad_s"])
            writer.writerows(commands)
        stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print("JSON:", stem.with_suffix(".json"), "\nCSV:", stem.with_suffix(".csv"), flush=True)
    return int(report["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
