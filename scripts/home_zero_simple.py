"""User-started joint-position move: MOVE_J 10%, 10 s recording.

No joint deviation, speed, settling or path segmentation thresholds.
Normal exit leaves the zero target and motors enabled. Ctrl+C requests
the documented damping emergency stop. Reset is an explicit CLI option.
"""
import argparse
import csv
import json
import math
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--record-diagnostics", action="store_true",
                        help="Record raw CAN, SDK send calls/states, console and 3 s after stopping")
    parser.add_argument("--target-deg",type=float,nargs=6,default=[0.]*6,
                        help="Explicit six-joint target in degrees (default: zero)")
    parser.add_argument("--reset-estop", action="store_true",
                        help="Operator has supported arm; reset removes motor power")
    args = parser.parse_args()
    target = [math.radians(v) for v in args.target_deg]
    props = subprocess.check_output(
        ["udevadm", "info", "--query=property", "--path=/sys/class/net/can0"], text=True)
    if "ID_SERIAL_SHORT=004F00374548570120353731" not in props.splitlines():
        raise RuntimeError("Piper H USB-CAN identity mismatch")
    robot = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.PIPER_H, firmeware_version=PiperFW.V189,
        interface="socketcan", channel="can0", bitrate=1000000,
        auto_connect=False, enable_check_can=True, timeout=.1))
    lock, pending, rows = threading.Lock(), set(), []
    received_at = {}
    driver_frames = []
    phase = "observe"
    recording = None
    view = SimpleNamespace(robot=robot,phase=phase)

    def observe(frame):
        if not frame.is_rx or frame.is_error_frame or frame.is_remote_frame:
            return
        received=time.monotonic()
        if len(frame.data)==8:received_at[frame.arbitration_id]=received
        if 0x261 <= frame.arbitration_id <= 0x266 and len(frame.data)==8:
            driver_frames.append({"timestamp_s":frame.timestamp,"phase":phase,
                                  "joint":frame.arbitration_id-0x260,"raw_hex":frame.data.hex()})
        if frame.arbitration_id not in (0x2A5, 0x2A6, 0x2A7) or len(frame.data) != 8:
            return
        pending.add(frame.arbitration_id)
        if len(pending) == 3:
            pending.clear()
            with lock:
                rows.append([received, frame.timestamp, phase,
                             *list(robot.get_joint_angles().msg)])

    def state():
        if robot.has_comm_error():
            raise RuntimeError(str(robot.get_comm_error()))
        with lock:
            if not rows:
                raise RuntimeError("No complete six-joint feedback")
            q = rows[-1][3:]
        status = robot.get_arm_status()
        return q, status.msg.to_dict() if status else None, robot.get_joints_enable_status_list()

    def motors_confirmed(expected, after):
        # enable() returns cached feedback, not an ACK for this request.
        messages = [robot.get_driver_states(j) for j in range(1, 7)]
        return all(m is not None and received_at.get(0x260+j,-math.inf) > after
                   and bool(m.msg.foc_status.driver_enable_status) == expected
                   for j,m in enumerate(messages,1))

    robot.get_context().register_parser_packet_fun(observe)
    report = {"speed_percent": 10, "target_rad": target,
              "record_seconds": 10, "calibration_changed": False,
              "motion_requested": False, "error": None}
    motion_requested = False
    def terminate(*_):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.record_diagnostics:
            from piper_oscbf.recording import TaskRecording,snapshot
            out=ROOT/'results'/('home_zero_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            out.mkdir()
            recording=TaskRecording(out,ROOT,extra_sources=('scripts/home_zero_simple.py',))
            recording.device=view
            report['diagnostic_directory']=str(out)
            print('DIAGNOSTIC_RESULT_DIR:',out,flush=True)
        robot.connect()
        if recording:recording.bind_connected_device(view)
        time.sleep(.6)
        q, status, enabled = state()
        report.update(initial_rad=q, initial_status=status, initial_enabled=enabled)
        if recording:recording.mark('connected',state=snapshot(view),target_deg=args.target_deg)
        print("当前角度 deg:", [round(math.degrees(x), 3) for x in q], flush=True)
        print("当前状态:", status, "使能:", enabled, flush=True)
        if args.execute:
            if status is None:
                raise RuntimeError("No arm status feedback")
            if int(status["arm_status"]) == 1:
                if not args.reset_estop:
                    raise RuntimeError("Software E-stop active; supported-arm recovery needs --reset-estop")
                print("恢复上轮软件急停：reset 会使电机失电；随后重新使能。", flush=True)
                phase = "reset"
                view.phase=phase
                reset_sent_at = time.monotonic()
                robot.reset()
                report["reset_sent"] = True
                reset_started = time.monotonic()
                while True:
                    time.sleep(.1)
                    q, status, enabled = state()
                    arm_message = robot.get_arm_status()
                    if (arm_message is not None and received_at.get(0x2A1,-math.inf) > reset_sent_at
                            and int(status["arm_status"]) == 0
                            and motors_confirmed(False, reset_sent_at)):
                        break
                    if time.monotonic()-reset_started > 5:
                        raise RuntimeError(f"Reset not completed: enabled={enabled}, status={status}")
                print("RESET_OK: 新反馈确认六轴已失能，软件急停已恢复。", flush=True)
            if status is None or int(status["arm_status"]) != 0:
                raise RuntimeError(f"Controller not ready: {status}")
            print("MOVE_J 10%，目标deg:",args.target_deg,"正在使能。Ctrl+C 电子急停。",flush=True)
            phase = "enable"
            view.phase=phase
            if recording:recording.mark('move_j_enable_start',state=snapshot(view),target_deg=args.target_deg)
            motion_requested = report["motion_requested"] = True
            robot.set_speed_percent(10)
            robot.move_j(target)
            enable_sent_at = enable_started = time.monotonic()
            while True:
                robot.enable()
                time.sleep(.1)
                q, status, enabled = state()
                if status is None or int(status["arm_status"]) != 0:
                    raise RuntimeError(f"Controller stopped/faulted during enable: {status}")
                if motors_confirmed(True, enable_sent_at):
                    break
                if time.monotonic()-enable_started > 5:
                    raise RuntimeError(f"Enable not confirmed: enabled={enabled}, status={status}")
            report["enable_confirmed"] = True
            report["enable_wait_s"] = time.monotonic()-enable_started
            print("ENABLE_OK: 六轴新反馈均为 True；开始记录 10 秒。", flush=True)
            phase = "home_zero" if not any(target) else "prepare_task"
            view.phase=phase
            if recording:recording.mark('move_j_recording_start',state=snapshot(view))
            robot.move_j(target)
            started, last_print = time.monotonic(), -1.0
            while time.monotonic()-started < 10:
                q, status, enabled = state()
                if status is None or int(status["arm_status"]) != 0:
                    raise RuntimeError(f"Controller stopped/faulted: {status}")
                if not all(enabled):
                    raise RuntimeError(f"Motor disabled during motion: {enabled}")
                robot.move_j(target)
                elapsed = time.monotonic()-started
                if elapsed-last_print >= .5:
                    print(f"{elapsed:5.1f}s q_deg={[round(math.degrees(x), 3) for x in q]} "
                          f"enabled={enabled} ctrl={status['ctrl_mode']} "
                          f"arm={status['arm_status']} motion={status['motion_status']}", flush=True)
                    last_print = elapsed
                time.sleep(.02)
            q, status, enabled = state()
            report.update(final_rad=q, final_deg=[math.degrees(x) for x in q],
                          final_status=status, final_enabled=enabled)
            print("RECORDING_FINISHED: 采集结束，未发送失能。末次使能反馈:", enabled, flush=True)
            print("末次实测角度 deg:", report["final_deg"], flush=True)
            if recording:recording.mark('move_j_recording_complete',state=snapshot(view))
        else:
            print("READ_ONLY: 未发送使能、运动或复位。", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report['exception_time_s']=time.time()
        report['exception_monotonic_s']=time.monotonic()
        report['traceback']=traceback.format_exc()
        if recording:recording.mark('exception_before_stop',state=snapshot(view),error=report['error'],traceback=report['traceback'])
        if motion_requested:
            try:
                robot.electronic_emergency_stop()
                report["stop_action"] = "damping_emergency_stop_requested"
                comm_error = robot.get_comm_error()
                report["stop_send_error"] = str(comm_error) if comm_error is not None else None
                print("已请求电子急停；SDK通信错误:", report["stop_send_error"],
                      "。调用返回不代表机械臂已收到；现场确认停止。", flush=True)
            except Exception as stop_error:
                report["stop_send_error"] = str(stop_error)
                print("急停未能发送，请使用物理急停。", flush=True)
        print("ERROR:", report["error"], flush=True)
    finally:
        if recording:
            phase=view.phase='post_stop_observation'
            recording.mark('passive_post_stop_started',state=snapshot(view),error=report['error'])
            print('仅继续采集停止后的状态3秒，不自动复位、重新使能或发送新目标。',flush=True)
            try:time.sleep(3.)
            except KeyboardInterrupt:pass
            report['post_stop_state']=snapshot(view)
            recording.finish()
        robot.disconnect()
        root = ROOT / "logs"
        root.mkdir(exist_ok=True)
        stem = root / ("home_zero_simple_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["monotonic_s", "can_timestamp_s", "phase"] + [f"q{i}_rad" for i in range(1, 7)])
            writer.writerows(rows)
        report["driver_frames"]=driver_frames
        stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        if recording:
            import shutil
            shutil.copy2(stem.with_suffix('.csv'),out/'feedback.csv')
            (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print('RESULT_DIR:',out,flush=True)
        print("JSON:", stem.with_suffix(".json"), "\nCSV:", stem.with_suffix(".csv"), flush=True)
    return int(report["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
