"""Piper H transport using the SDK API verified in cpv_axes_demo.py.

Only the calling thread sends commands. The SDK receive callback assembles
fresh complete joint states; it never sends motion commands.
"""
import math
import subprocess
import threading
import time

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
from piper_oscbf.pacing import GroupPacer


class MotorEnableUnconfirmed(RuntimeError):
    """Enable feedback is absent/disabled without an explicit driver fault bit."""


class PiperDevice:
    def __init__(self, cpv_period_s=None):
        props = subprocess.check_output([
            "udevadm", "info", "--query=property", "--path=/sys/class/net/can0"], text=True)
        if "ID_SERIAL_SHORT=004F00374548570120353731" not in props.splitlines():
            raise RuntimeError("Piper H USB-CAN identity mismatch")
        self.robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.PIPER_H, firmeware_version=PiperFW.V189,
            interface="socketcan", channel="can0", bitrate=1000000,
            auto_connect=False, enable_check_can=True, timeout=.05))
        self.condition, self.pending = threading.Condition(), {}
        self.latest, self.sequence = None, 0
        self.received_at = {}
        self.phase, self.commanded = "observe", [0.0]*6
        self.feedback, self.commands = [], []
        self.driver_feedback = []
        self.last_driver_check = None
        self.driver_feedback_pauses = []
        self.driver_pause_s = 0.
        self.control_started = False
        # Existing independent plane experiments retain their own scheduler.
        # The complete task explicitly opts into continuous group pacing.
        self.cpv_pacer = GroupPacer(cpv_period_s) if cpv_period_s is not None else None
        self.robot.get_context().register_parser_packet_fun(self._receive)
        self.robot.set_auto_set_motion_mode_enabled(False)

    def _receive(self, frame):
        received = time.monotonic()
        if (frame.is_rx and not frame.is_remote_frame and not frame.is_error_frame
                and len(frame.data) == 8):
            self.received_at[frame.arbitration_id] = received
        if (frame.is_rx and not frame.is_remote_frame and not frame.is_error_frame
                and 0x261 <= frame.arbitration_id <= 0x266 and len(frame.data) == 8):
            self.driver_feedback.append({"monotonic_s": received,
                "can_timestamp_s": frame.timestamp, "phase": self.phase,
                "joint": frame.arbitration_id-0x260, "status_byte": int(frame.data[5]),
                "enabled": bool(frame.data[5] & 0x40), "raw_hex": frame.data.hex()})
        if (not frame.is_rx or frame.is_remote_frame or frame.is_error_frame or
                frame.arbitration_id not in (0x2A5, 0x2A6, 0x2A7) or len(frame.data) != 8):
            return
        self.pending[frame.arbitration_id] = (received, frame.timestamp)
        if len(self.pending) != 3:
            return
        stamp = min(v[1] for v in self.pending.values())
        mono = min(v[0] for v in self.pending.values())
        self.pending.clear()
        self.sequence += 1
        row = {"sequence": self.sequence, "monotonic_s": mono, "can_timestamp_s": stamp,
               "phase": self.phase}
        motors = [self.robot.get_motor_states(j) for j in range(1, 7)]
        for j, (q, m) in enumerate(zip(self.robot.get_joint_angles().msg, motors), 1):
            row[f"q{j}_rad"] = float(q)
            row[f"command{j}_rad_s"] = self.commanded[j-1]
            row[f"motor{j}_position_rad"] = m.msg.position if m else math.nan
            row[f"motor{j}_velocity_rad_s"] = m.msg.velocity if m else math.nan
            row[f"motor{j}_timestamp_s"] = m.timestamp if m else math.nan
        with self.condition:
            self.latest = row
            self.feedback.append(row)
            self.condition.notify_all()

    def connect(self):
        self.robot.connect()
        time.sleep(.6)

    def state(self, after=None, enabled=True):
        with self.condition:
            if after is not None:
                self.condition.wait_for(lambda: self.latest is not None and
                                        self.latest["sequence"] > after, timeout=.2)
            row = self.latest.copy() if self.latest else None
        if self.robot.has_comm_error():
            raise RuntimeError(str(self.robot.get_comm_error()))
        if (row is None or time.monotonic()-row["monotonic_s"] > .2 or
                (after is not None and row["sequence"] <= after)):
            raise RuntimeError("Fresh complete joint feedback interrupted")
        status = self.robot.get_arm_status()
        if status is None or self.message_age(0x2A1) > .2:
            raise RuntimeError("Arm status feedback interrupted")
        if int(status.msg.arm_status) != 0:
            raise RuntimeError(f"Controller stopped/faulted: {status.msg}")
        if enabled:
            drivers = [self.robot.get_driver_states(j) for j in range(1, 7)]
            checks = [{"joint": j, "age_s": self.message_age(0x260+j) if m else None,
                       "enabled": bool(m.msg.foc_status.driver_enable_status) if m else None}
                      for j,m in enumerate(drivers,1)]
            bad = [m for m in checks if m["age_s"] is None or m["age_s"] >= .2 or not m["enabled"]]
            if bad:
                for check,message in zip(checks,drivers):
                    check["status"] = message.msg.to_dict() if message else None
                self.last_driver_check = {"time_s": time.time(), "phase": self.phase, "drivers": checks,
                                          "age_clock": "monotonic_receive"}
                fault = any(m["status"] is not None and any(v for k,v in m["status"]["foc_status"].items()
                            if k!="driver_enable_status") for m in bad)
                error_type = RuntimeError if fault else MotorEnableUnconfirmed
                raise error_type(f"Motor enable feedback failed: {bad}")
        return row, status.msg.to_dict()

    def message_age(self, frame_id):
        """Age since this callback received the CAN message, in one clock domain.

        SocketCAN's epoch timestamp remains in logs; a WSL wall-clock correction
        must not turn an already received frame into a stale one.
        """
        return time.monotonic() - self.received_at.get(frame_id, -math.inf)

    @staticmethod
    def angles(row):
        return [row[f"q{j}_rad"] for j in range(1, 7)]

    def enable_at(self, current):
        self.control_started = True
        self.phase = "enable"
        self.robot.set_speed_percent(10)
        self.robot.set_motion_mode("j")
        self.robot.move_j(list(current))
        stamp = start = time.monotonic()
        while True:
            self.robot.enable()
            time.sleep(.1)
            self.state(enabled=False)
            messages = [self.robot.get_driver_states(j) for j in range(1, 7)]
            if all(m is not None and self.received_at.get(0x260+j, -math.inf) > stamp and
                   m.msg.foc_status.driver_enable_status for j,m in enumerate(messages,1)):
                return
            if time.monotonic()-start > 5:
                raise RuntimeError("Enable feedback not confirmed")

    def position(self, target, phase):
        self.phase, self.commanded = phase, [0.0]*6
        self.robot.set_speed_percent(10)
        self.robot.set_motion_mode("j")
        start = time.monotonic()
        while time.monotonic()-start < 5:
            self.state()
            self.robot.move_j(list(target))
            time.sleep(.02)
        row, status = self.state()
        if int(status["mode_feedback"]) != 1 or int(status["motion_status"]) != 0:
            raise RuntimeError("Controller has not reported MOVE_J arrival")
        print(phase, "q_deg=", [round(math.degrees(x), 3) for x in self.angles(row)], flush=True)
        return row

    def wait_for_velocity_slot(self):
        if self.cpv_pacer is not None:
            self.cpv_pacer.wait()

    def velocity(self, values, *, immediate=False):
        values = [float(v) for v in values]
        # Same rounding as SDK's sp scale (1e3). These are model-convention
        # commands passed to SDK; SDK internally compensates J2-J5 wire signs.
        self.commanded = [round(v*1000)/1000 for v in values]
        if not immediate:
            self.wait_for_velocity_slot()
        start = time.monotonic()
        previous = self.commands[-1] if self.commands else None
        timing = (self.cpv_pacer.begin(start, immediate=immediate) if self.cpv_pacer is not None else {
            'pacing_period_s': None, 'pacing_bypassed': immediate, 'scheduled_start_s': None,
            'pacing_wait_s': 0., 'deadline_lateness_s': None,
            'interval_from_previous_start_s': start-previous['send_start_s'] if previous else None,
            'gap_from_previous_end_s': start-previous['send_end_s'] if previous else None})
        completed, error = 0, None
        try:
            for j, v in enumerate(self.commanded, 1):
                self.robot.move_cpv_vel(j, v)
                completed += 1
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            end = time.monotonic()
            if self.cpv_pacer is not None:
                self.cpv_pacer.end(end)
            self.commands.append({"send_start_s": start, "send_end_s": end, "phase": self.phase,
                **timing, 'completed_joints': completed, 'send_error': error,
                **{f"u{j}_rad_s": v for j, v in enumerate(self.commanded, 1)}})
        return self.commanded[:]

    def zero(self, duration=1.0):
        self.phase = "zero_velocity"
        start = time.monotonic()
        while time.monotonic()-start < duration:
            self.wait_for_velocity_slot()
            self.state()
            self.velocity([0.0]*6)
            if self.cpv_pacer is None:
                time.sleep(.01)

    def emergency_stop(self):
        errors = []
        try:
            self.velocity([0.0]*6, immediate=True)
        except Exception as exc:
            errors.append(str(exc))
        try:
            self.robot.electronic_emergency_stop()
            print("已请求电子急停，可能下沉；现场确认停止，必要时使用物理急停。", flush=True)
        except Exception as exc:
            errors.append(str(exc))
            print("电子急停发送失败，请使用物理急停。", flush=True)
        return errors

    def disconnect(self):
        self.robot.disconnect()
