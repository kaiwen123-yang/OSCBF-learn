"""Operator-started Piper H CPV experiment; default invocation only receives.

MOVE_J 10%: measured q + [0,10,-10,0,10,0] deg; then sequential J1-J5
velocity pulses +0.02/0/-0.02/0 rad/s for 1.5/0.75/1.5/0.75 s.
All six joints receive explicit velocity references throughout CPV; no CPV
position commands, parameter writes, gripper commands, reset or calibration.
Finish in MOVE_J at the measured initial q and remain enabled.
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
JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)
SPEED = 0.02
DT = 0.01


def degrees(q):
    return [round(math.degrees(v), 4) for v in q]


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
        channel="can0", interface="socketcan", bitrate=1000000,
        auto_connect=False, enable_check_can=True, timeout=.05))
    lock, pending, rows, commands = threading.Lock(), {}, [], []
    phase, commanded = "observe", [0.0]*6
    control_started = False
    report = {"time": datetime.now().astimezone().isoformat(), "execute": args.execute,
              "robot": "piper_h", "sdk_profile": "v189", "channel": "can0",
              "move_j_percent": 10, "pulse_velocity_rad_s": SPEED,
              "phases": [], "completed_sequence": False, "error": None,
              "note": "Measured feedback, SDK input signs; no automatic sign calibration. "
                      "CPV has velocity commands on all six axes, no position writes."}

    def observe(frame):
        if (not frame.is_rx or frame.is_error_frame or frame.is_remote_frame or
                frame.arbitration_id not in JOINT_IDS or len(frame.data) != 8):
            return
        pending[frame.arbitration_id] = (time.monotonic(), frame.timestamp)
        if len(pending) != 3:
            return
        oldest = min(x[0] for x in pending.values())
        first_stamp = min(x[1] for x in pending.values())
        last_stamp = max(x[1] for x in pending.values())
        pending.clear()
        motors = [robot.get_motor_states(j) for j in range(1, 7)]
        # Preserve per-source timestamps: these are not simultaneous samples.
        row = {"monotonic_s": oldest, "can_first_s": first_stamp,
               "can_last_s": last_stamp, "phase": phase}
        for j, (q, motor) in enumerate(zip(robot.get_joint_angles().msg, motors), 1):
            row[f"q{j}_rad"] = q
            row[f"command{j}_rad_s"] = commanded[j-1]
            row[f"motor{j}_position_rad"] = motor.msg.position if motor else math.nan
            row[f"motor{j}_velocity_rad_s"] = motor.msg.velocity if motor else math.nan
            row[f"motor{j}_timestamp_s"] = motor.timestamp if motor else math.nan
        flange = robot.get_flange_pose()
        if flange is not None:
            for k, value in enumerate(flange.msg):
                row[f"flange_{k}"] = value
            row["flange_timestamp_s"] = flange.timestamp
        else:
            for k in range(6):
                row[f"flange_{k}"] = math.nan
            row["flange_timestamp_s"] = math.nan
        with lock:
            rows.append(row)

    def state(require_enabled=True):
        if robot.has_comm_error():
            raise RuntimeError(str(robot.get_comm_error()))
        with lock:
            row = rows[-1].copy() if rows else None
        if row is None or time.monotonic()-row["monotonic_s"] > .2:
            raise RuntimeError("Complete joint feedback interrupted")
        message = robot.get_arm_status()
        if message is None or time.time()-message.timestamp > .2:
            raise RuntimeError("Arm status feedback interrupted")
        if int(message.msg.arm_status) != 0:
            raise RuntimeError(f"Controller stopped/faulted: {message.msg}")
        drivers = [robot.get_driver_states(j) for j in range(1, 7)]
        if require_enabled and not all(
                m is not None and time.time()-m.timestamp < .2 and
                m.msg.foc_status.driver_enable_status for m in drivers):
            raise RuntimeError("Six enabled motors not confirmed by fresh feedback")
        return row, message.msg.to_dict()

    def snapshot(row):
        return {"q_rad": [row[f"q{j}_rad"] for j in range(1, 7)],
                "motor_position_rad": [row[f"motor{j}_position_rad"] for j in range(1, 7)],
                "motor_velocity_rad_s": [row[f"motor{j}_velocity_rad_s"] for j in range(1, 7)]}

    def send_velocity(values):
        nonlocal commanded
        commanded = list(values)
        started = time.monotonic()
        for j, value in enumerate(values, 1):
            robot.move_cpv_vel(j, value)
        commands.append([started, time.monotonic(), phase, *values])

    def position_phase(name, target):
        nonlocal phase, commanded
        phase, commanded = name, [0.0]*6
        robot.set_speed_percent(10)
        robot.set_motion_mode("j")
        started = time.monotonic()
        while time.monotonic()-started < 5:
            state()
            robot.move_j(target)
            time.sleep(.02)
        row, status = state()
        report[name] = {**snapshot(row), "status": status, "target_rad": target}
        print(f"{name}: q_deg={degrees(snapshot(row)['q_rad'])} "
              f"mode={status['mode_feedback']} motion={status['motion_status']}", flush=True)
        # This is the controller's arrival flag, not a user-space angular threshold.
        if int(status["mode_feedback"]) != 1 or int(status["motion_status"]) != 0:
            raise RuntimeError("Controller has not reported MOVE_J arrival; CPV sequence not continued")

    def velocity_phase(name, joint, value, seconds):
        nonlocal phase
        phase = name
        first, _ = state()
        vector = [0.0]*6
        if joint:
            vector[joint-1] = value
        started, deadline = time.monotonic(), time.monotonic()
        while time.monotonic()-started < seconds:
            _, status = state()
            if name != "zero_all_before" and int(status["mode_feedback"]) != 5:
                raise RuntimeError(f"CPV mode not confirmed: {status}")
            send_velocity(vector)
            deadline += DT
            time.sleep(max(0., deadline-time.monotonic()))
        send_velocity([0.0]*6)
        last, status = state()
        elapsed = last["monotonic_s"]-first["monotonic_s"]
        start_snapshot, end_snapshot = snapshot(first), snapshot(last)
        delta = [b-a for a, b in zip(start_snapshot["q_rad"], end_snapshot["q_rad"])]
        motor_delta = [b-a for a, b in zip(start_snapshot["motor_position_rad"],
                                         end_snapshot["motor_position_rad"])]
        item = {"phase": name, "joint": joint, "sdk_command_rad_s": value,
                "elapsed_s": elapsed, "start": start_snapshot, "end": end_snapshot,
                "joint_delta_deg": degrees(delta), "motor_delta_deg": degrees(motor_delta),
                "status": status}
        if joint:
            item["mean_joint_velocity_rad_s"] = delta[joint-1]/elapsed
            item["response_gain"] = delta[joint-1]/(elapsed*value) if value else None
            print(f"{name}: SDK={value:+.3f} rad/s, "
                  f"关节Δ={math.degrees(delta[joint-1]):+.3f}°, "
                  f"电机Δ={math.degrees(motor_delta[joint-1]):+.3f}°, "
                  f"关节均速={delta[joint-1]/elapsed:+.5f} rad/s", flush=True)
        else:
            print(f"{name}: 六轴变化 deg={degrees(delta)}", flush=True)
        report["phases"].append(item)

    def terminate(*_):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    robot.get_context().register_parser_packet_fun(observe)
    try:
        robot.connect()
        time.sleep(.6)
        first, status = state(require_enabled=False)
        q0 = snapshot(first)["q_rad"]
        prepared = [q+math.radians(d) for q, d in zip(q0, (0, 10, -10, 0, 10, 0))]
        report.update(initial=snapshot(first), initial_status=status, prepared_target_rad=prepared)
        print("当前实测 deg:", degrees(q0), flush=True)
        print("准备目标 deg:", degrees(prepared), "MOVE_J 10%", flush=True)
        print("J1-J5 逐轴 +0.02(1.5s) / 0(0.75s) / -0.02(1.5s) / 0(0.75s)。", flush=True)
        print("正常结束：MOVE_J 返回本次初始位置并保持使能；Ctrl+C：零速后电子急停，可能下沉。", flush=True)
        for j, target in enumerate(prepared, 1):
            lo, hi = robot.get_config()["joint_limits"][f"joint{j}"]
            if not lo <= target <= hi or not lo <= q0[j-1] <= hi:
                raise RuntimeError(f"J{j}: initial or relative target outside official SDK limits")
        if not args.execute:
            print("READ_ONLY: 仅接收和预览；执行需要 --execute。", flush=True)
        else:
            robot.set_auto_set_motion_mode_enabled(False)
            control_started = True
            robot.set_speed_percent(10)
            robot.set_motion_mode("j")
            robot.move_j(q0)
            enable_time, started = time.time(), time.monotonic()
            while True:
                robot.enable()
                time.sleep(.1)
                state(require_enabled=False)
                drivers = [robot.get_driver_states(j) for j in range(1, 7)]
                if all(m is not None and m.timestamp > enable_time and
                       m.msg.foc_status.driver_enable_status for m in drivers):
                    break
                if time.monotonic()-started > 5:
                    raise RuntimeError("Enable feedback not confirmed")
            position_phase("prepare", prepared)
            robot.set_motion_mode("cpv")
            velocity_phase("zero_all_before", 0, 0., 1.0)
            for joint in range(1, 6):
                velocity_phase(f"j{joint}_positive", joint, SPEED, 1.5)
                velocity_phase(f"j{joint}_zero_after_positive", joint, 0., .75)
                velocity_phase(f"j{joint}_negative", joint, -SPEED, 1.5)
                velocity_phase(f"j{joint}_zero_after_negative", joint, 0., .75)
            velocity_phase("zero_all_final", 0, 0., 1.0)
            position_phase("return_initial", q0)
            report["completed_sequence"] = True
            print("CPV_AXES_SEQUENCE_FINISHED: 数据已采集，已返回初始目标，保持使能。", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if control_started:
            # Manual mode management stays enabled; zero packets cannot switch
            # a failed MOVE_J into CPV during cleanup.
            try:
                send_velocity([0.0]*6)
            except Exception as stop_error:
                report["zero_send_error"] = str(stop_error)
            try:
                robot.electronic_emergency_stop()
                print("已请求电子急停，可能下沉；现场确认停止，必要时物理急停。", flush=True)
            except Exception as stop_error:
                report["estop_send_error"] = str(stop_error)
                print("电子急停发送失败，请使用现场物理急停。", flush=True)
        print("ERROR:", report["error"], flush=True)
    finally:
        robot.disconnect()
        stem = ROOT / "logs" / ("cpv_axes_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if rows:
            with stem.with_suffix(".csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with Path(str(stem)+"_commands.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["send_start_s", "send_end_s", "phase"]+[f"u{j}_rad_s" for j in range(1, 7)])
            writer.writerows(commands)
        stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print("JSON:", stem.with_suffix(".json"), "\nCSV:", stem.with_suffix(".csv"), flush=True)
    return int(report["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
