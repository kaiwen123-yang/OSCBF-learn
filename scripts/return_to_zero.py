"""Operator-started, measured-feedback joint homing; never recalibrates zero.

Default: query and preview only. --execute requires the local HOME ZERO prompt.
No ROS, CPV, gripper, reset, calibration, limit-writing or disable calls.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_COMMIT = "d90e234ba8e408cfa964b38aa9120b3131fc9f85"
SERIAL = "004F00374548570120353731"
RAD = math.pi / 180
TOL = 0.15 * RAD


def zero_path(q, bounds):
    """All commanded endpoints are legal. Only the observed J2/J3 start is exempt."""
    if len(q) != 6 or not all(math.isfinite(x) for x in q):
        raise ValueError("Invalid six-joint state")
    # This entry is scoped to the reviewed posture, not an arbitrary home path.
    reference = [0.372, -2.706, 0.564, 0.544, 29.92, 0.0]
    if any(abs(x / RAD - r) > 5 for x, r in zip(q, reference)):
        raise ValueError("Posture differs by >5 deg from reviewed posture; re-review path")
    for i, (x, (lo, hi)) in enumerate(zip(q, bounds)):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo <= 0 <= hi):
            raise ValueError(f"J{i+1}: zero is not in valid controller bounds")
        if lo <= x <= hi:
            continue
        if i == 1 and lo == 0 and -3.5 * RAD <= x < 0:
            continue
        if i == 2 and hi == 0 and 0 < x <= 1.5 * RAD:
            continue
        raise ValueError(f"J{i+1}: unreviewed out-of-bounds start")
    first = list(q)
    first[1] = first[2] = 0.0
    if max(abs(a-b) for a, b in zip(q, first)) > 3.5 * RAD:
        raise ValueError("Initial J2/J3 correction exceeds 3.5 deg")
    path = [first]
    while any(abs(x) > 1e-10 for x in path[-1]):
        path.append([math.copysign(max(abs(x) - 2 * RAD, 0.0), x)
                     for x in path[-1]])
    for target in path:
        if any(not lo <= x <= hi for x, (lo, hi) in zip(target, bounds)):
            raise ValueError("A target lies outside controller limits")
    return path


class Feedback:
    def __init__(self, robot):
        self.robot = robot
        self.lock = threading.Lock()
        self.pending = {}
        self.latest = None
        self.drivers = {}
        self.status = None
        self.rows = []
        self.phase = "preflight"

    def observe(self, frame):
        if not frame.is_rx or frame.is_remote_frame or frame.is_error_frame:
            return
        fid, now = frame.arbitration_id, time.monotonic()
        if fid in (0x2A5, 0x2A6, 0x2A7) and len(frame.data) == 8:
            self.pending[fid] = (frame.timestamp, now)
            if len(self.pending) != 3:
                return
            first, last = min(v[0] for v in self.pending.values()), max(v[0] for v in self.pending.values())
            first_mono,last_mono = min(v[1] for v in self.pending.values()),max(v[1] for v in self.pending.values())
            self.pending.clear()
            if last_mono-first_mono > .020 or not 0 <= now-first_mono <= .100:
                return
            msg = self.robot.get_joint_angles()
            if msg is None or len(msg.msg) != 6 or not all(math.isfinite(x) for x in msg.msg):
                return
            with self.lock:
                self.latest = (first_mono, first, list(msg.msg))
                self.rows.append([now, first, self.phase, *msg.msg])
        elif 0x261 <= fid <= 0x266:
            j = fid - 0x260
            msg = self.robot.get_driver_states(j)
            if msg is not None:
                with self.lock:
                    self.drivers[j] = (now, msg.timestamp, msg.msg.to_dict())
        elif fid == 0x2A1:
            msg = self.robot.get_arm_status()
            if msg is not None:
                with self.lock:
                    self.status = (now, msg.timestamp, msg.msg.to_dict())

    def read(self, required_enabled=None):
        now = time.monotonic()
        with self.lock:
            if self.latest is None or now-self.latest[0] > .100:
                raise RuntimeError("Complete joint feedback missing/stale (>100 ms)")
            messages = [self.drivers.get(j) for j in range(1, 7)] + [self.status]
            if any(m is None or now-m[0] > .200 for m in messages):
                raise RuntimeError("Driver/arm feedback missing/stale (>200 ms)")
            q = self.latest[2][:]
            drivers = [m[2] for m in messages[:6]]
            status = self.status[2]
        enabled = []
        for j, driver in enumerate(drivers, 1):
            foc = driver["foc_status"]
            enabled.append(bool(foc["driver_enable_status"]))
            faults = [key for key, val in foc.items() if key != "driver_enable_status" and val]
            if faults:
                raise RuntimeError(f"J{j} driver faults: {faults}")
        if int(status["arm_status"]) != 0 or any(status["err_status"].values()):
            raise RuntimeError(f"Arm fault: {status}")
        if required_enabled is not None and enabled != [required_enabled] * 6:
            raise RuntimeError(f"Unexpected joint enable state: {enabled}")
        if self.robot.has_comm_error():
            raise RuntimeError(str(self.robot.get_comm_error()))
        return q, enabled, status


def counters():
    base = Path("/sys/class/net/can0/statistics")
    return {k: int((base / k).read_text()) for k in
            ("rx_packets", "rx_dropped", "rx_errors", "tx_packets", "tx_errors")}


def verify_sdk():
    import pyAgxArm
    repo = ROOT / "src/pyAgxArm"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != SDK_COMMIT:
        raise RuntimeError("SDK commit changed; re-audit required")
    subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", "pyAgxArm"],
                   check=True, stdout=subprocess.DEVNULL)
    installed = Path(pyAgxArm.__file__).parent
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "pyAgxArm"], text=True).split("\0")
    for name in tracked:
        # Upstream tracks .log/__init__.py, which setuptools does not package.
        if not name.endswith(".py") or any(part.startswith(".") for part in Path(name).parts):
            continue
        src = repo / name
        dst = installed / src.relative_to(repo / "pyAgxArm")
        if not dst.is_file() or hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
            raise RuntimeError(f"Installed SDK differs from audited source: {src}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
    verify_sdk()
    props = subprocess.check_output(
        ["udevadm", "info", "--query=property", "--path=/sys/class/net/can0"], text=True)
    if f"ID_SERIAL_SHORT={SERIAL}" not in props.splitlines():
        raise RuntimeError("USB-CAN identity mismatch")
    ROOT.joinpath("logs").mkdir(exist_ok=True)
    lease = (ROOT / "logs/hardware_command.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = create_agx_arm_config(
        robot=ArmModel.PIPER_H, firmeware_version=PiperFW.V189,
        interface="socketcan", channel="can0", bitrate=1000000,
        auto_connect=False, enable_check_can=True, timeout=.05)
    robot = AgxArmFactory.create_arm(config)
    feedback = Feedback(robot)
    robot.get_context().register_parser_packet_fun(feedback.observe)
    report = {"time": datetime.now().astimezone().isoformat(), "sdk_commit": SDK_COMMIT,
              "mode": "execute" if args.execute else "preview", "speed_percent": 1,
              "target_rad": [0.0]*6, "calibration_changed": False,
              "success": False, "enabled_by_script": False, "error": None,
              "stages": []}
    before = counters()
    enable_attempted = False
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        robot.connect()
        time.sleep(.6)
        q0, _, _ = feedback.read(required_enabled=False)
        time.sleep(.4)
        q1, _, _ = feedback.read(required_enabled=False)
        if max(abs(a-b) for a, b in zip(q0, q1)) > TOL:
            raise RuntimeError("Disabled arm is not stationary; do not enable")
        fw = robot.get_firmware(timeout=2)
        report["firmware"] = fw
        if fw is None or fw["software_version"].strip() != "S-V1.9-1":
            raise RuntimeError("Firmware differs from S-V1.9-1")
        bounds = []
        for j in range(1, 7):
            feedback.read(required_enabled=False)
            msg = robot.get_joint_angle_vel_limits(j, timeout=1)
            if msg is None:
                raise RuntimeError(f"J{j} limit query failed")
            bounds.append((msg.msg.min_angle_limit, msg.msg.max_angle_limit))
        q0, _, _ = feedback.read(required_enabled=False)
        path = zero_path(q0, bounds)
        report.update(initial_rad=q0, controller_bounds_rad=bounds, planned_targets_rad=path)
        print("实测角度 deg:", [round(x/RAD, 3) for x in q0], flush=True)
        print("各轴回零行程 deg:", [round(-x/RAD, 3) for x in q0], flush=True)
        print("首段目标 deg:", [round(x/RAD, 3) for x in path[0]], flush=True)
        print(f"后续单段 <=2 deg，共 {len(path)} 段；MOVE_J 速度比例 1%。", flush=True)
        print("本入口不含场景碰撞检测。需现场确认整个回零路径无障碍。", flush=True)
        print("正常结束保持使能；异常/Ctrl+C 发电子急停（阻尼停机，可能下沉）。", flush=True)
        if not args.execute:
            report["preview_ok"] = True
            print("PREVIEW_OK: 未发送使能、运动或设置指令。", flush=True)
        else:
            if input("确认现场路径、急停且无其他实机控制程序；输入 HOME ZERO 开始: ").strip() != "HOME ZERO":
                raise RuntimeError("Operator cancelled before enable")
            q, _, _ = feedback.read(required_enabled=False)
            if max(abs(a-b) for a, b in zip(q, q0)) > TOL:
                raise RuntimeError("Arm moved after preview; rerun to generate new targets")
            current = counters()
            delta = {k: current[k]-before[k] for k in before}
            if any(delta[k] for k in ("rx_dropped", "rx_errors", "tx_errors")):
                raise RuntimeError(f"CAN preflight errors/drops: {delta}")
            # Finite, legal, small first target is installed WHILE ALL MOTORS ARE OFF.
            # No unknown stale target is deliberately enabled. Speed is set first.
            robot.set_speed_percent(1)
            robot.move_j(path[0])
            time.sleep(.1)
            feedback.read(required_enabled=False)
            feedback.phase = "enable"
            enable_attempted = True
            enable_start = time.monotonic()
            robot.enable()
            last_enable = enable_start
            while True:
                q, enabled, _ = feedback.read()
                if max(abs(a-b) for a, b in zip(q, q0)) > 4 * RAD:
                    raise RuntimeError("Unexpected displacement during enable")
                if all(enabled):
                    report["enabled_by_script"] = True
                    break
                if time.monotonic()-enable_start > 2:
                    raise RuntimeError("Enable feedback did not confirm all six joints")
                if time.monotonic()-last_enable > .25:
                    robot.enable()
                    last_enable = time.monotonic()
                time.sleep(.01)
            motion_start = time.monotonic()
            last_window = (motion_start, q[:])
            for n, target in enumerate(path, 1):
                feedback.phase = f"move_{n}"
                start_q, _, _ = feedback.read(required_enabled=True)
                if max(abs(a-b) for a, b in zip(start_q, target)) > (3.7 if n == 1 else 2.3)*RAD:
                    raise RuntimeError("Measured state is too far from next small target")
                robot.move_j(target)
                started, settled_since, last_send = time.monotonic(), None, time.monotonic()
                while True:
                    now = time.monotonic()
                    q, _, status = feedback.read(required_enabled=True)
                    if int(status["ctrl_mode"]) != 1 or int(status["mode_feedback"]) != 1:
                        if now-started > .2:
                            raise RuntimeError("Controller did not confirm CAN / MOVE_J mode")
                    if now-started > 12 or now-motion_start > 240:
                        raise RuntimeError("Motion/settling timeout")
                    for i, x in enumerate(q):
                        if not min(start_q[i], target[i])-.5*RAD <= x <= max(start_q[i], target[i])+.5*RAD:
                            raise RuntimeError(f"J{i+1}: deviated from current segment")
                    if now-last_window[0] >= .1:
                        speed = max(abs(a-b)/(now-last_window[0]) for a, b in zip(q, last_window[1]))
                        if speed > 8*RAD:
                            raise RuntimeError("Measured speed exceeds 8 deg/s supervision threshold")
                        last_window = (now, q[:])
                        current = counters()
                        if any(current[k] != before[k] for k in ("rx_dropped", "rx_errors", "tx_errors")):
                            raise RuntimeError("CAN errors/drops during motion")
                    err = max(abs(a-b) for a, b in zip(q, target))
                    settled_since = (settled_since or now) if err <= TOL else None
                    if settled_since is not None and now-settled_since >= .5:
                        break
                    if now-last_send >= .1:
                        robot.move_j(target)  # Repeats only this bounded position endpoint.
                        last_send = now
                    time.sleep(.01)
                report["stages"].append({"stage": n, "target_rad": target, "measured_rad": q,
                                         "duration_s": time.monotonic()-started})
                print(f"阶段 {n}/{len(path)} 实测 deg: {[round(x/RAD, 3) for x in q]}", flush=True)
            report.update(success=True, final_rad=q, final_max_error_deg=max(abs(x)/RAD for x in q),
                          final_state="enabled_position_hold", stop_action="none")
            print("RETURN_ZERO_OK: 六轴反馈已到 0±0.15 deg，保持使能；未修改零位。", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if enable_attempted:
            # This is the documented damping stop, not an unverified position hold.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            report["stop_action"] = "electronic_emergency_stop_attempted"
            for _ in range(3):
                try:
                    robot.electronic_emergency_stop()
                    time.sleep(.02)
                except Exception as stop_error:
                    report["stop_send_error"] = str(stop_error)
            print("已尝试电子急停；可能下沉。现场确认停止，必要时立即使用物理急停。", flush=True)
            # No automatic reset, disable, or continuation after a stop.
        print("STOPPED:", report["error"], flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        try:
            robot.disconnect()
        except Exception as exc:
            report["disconnect_error"] = str(exc)
        try:
            current = counters()
            report["counter_delta"] = {k: current[k]-before[k] for k in before}
        except Exception as exc:
            report["counter_read_error"] = str(exc)
        stem = ROOT / "logs" / ("return_zero_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["monotonic_s", "can_timestamp_s", "phase"] + [f"q{j}_rad" for j in range(1, 7)])
            writer.writerows(feedback.rows)
        stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print("JSON:", stem.with_suffix(".json"), "\nCSV:", stem.with_suffix(".csv"), flush=True)
        lease.close()
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
