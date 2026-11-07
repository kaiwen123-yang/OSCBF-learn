"""Focused fake-clock check of the real six-axis send/normal-stop call path.

No hardware connection, ROS, GUI, or motion. This intentionally fails before
the pacing fix because a normal command is immediately followed by zero().
"""
import json
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import piper_oscbf.device as transport


class Clock:
    def __init__(self):
        self.now = 1.0
    def monotonic(self):
        return self.now
    def sleep(self, delay):
        assert delay >= 0
        self.now += delay


class Robot:
    def __init__(self, clock):
        self.clock, self.calls, self.estops = clock, [], 0
    def move_cpv_vel(self, joint, value):
        self.calls.append((self.clock.now, joint, value))
        self.clock.now += .00002
    def electronic_emergency_stop(self):
        self.estops += 1


def main():
    clock = Clock()
    original_time = transport.time
    transport.time = clock
    try:
        device = transport.PiperDevice.__new__(transport.PiperDevice)
        device.robot = Robot(clock)
        device.commands, device.commanded, device.phase = [], [0.] * 6, 'motion'
        device.state = lambda **_: ({}, {})
        if (ROOT / 'piper_oscbf/pacing.py').exists():
            from piper_oscbf.pacing import GroupPacer
            device.cpv_pacer = GroupPacer(.01, clock=clock.monotonic, sleep=clock.sleep)
        else:
            device.cpv_pacer = None
        device.velocity([.01049] * 6)
        device.zero(.025)
        device.phase = 'next_stage'
        device.velocity([.02] * 6)
        normal = list(device.commands)
        intervals = [b['send_start_s'] - a['send_start_s'] for a, b in zip(normal, normal[1:])]
        assert min(intervals) >= .01 - 1e-12, f'NORMAL_GROUP_BURST: min start interval {min(intervals)*1000:.6f} ms'
        assert all(j == (i % 6) + 1 for i, (_, j, _) in enumerate(device.robot.calls))
        assert all(value == .010 for _, _, value in device.robot.calls[:6])
        # A missed deadline may yield one late group, never a catch-up burst.
        clock.now += .035
        device.velocity([.03] * 6)
        device.velocity([.03] * 6)
        assert device.commands[-1]['send_start_s'] - device.commands[-2]['send_start_s'] >= .01 - 1e-12
        # Stop uses the actual emergency path and must bypass normal pacing.
        stop_requested = clock.now
        with contextlib.redirect_stdout(io.StringIO()):
            errors = device.emergency_stop()  # Robot here is a fake, never CAN.
        assert not errors and device.robot.estops == 1
        assert device.commands[-1]['send_start_s'] - stop_requested < 1e-12
        assert clock.now - stop_requested < .001
        assert device.commands[-1]['pacing_bypassed']
        device.phase = 'after_stop_check_only'
        device.velocity([0.] * 6)
        assert device.commands[-1]['send_start_s'] - device.commands[-2]['send_start_s'] >= .01 - 1e-12
        print(json.dumps({'status': 'passed', 'hardware_connected': False,
            'normal_start_intervals_ms': [v * 1000 for v in intervals],
            'normal_then_zero_then_next_stage': 'passed', 'no_catch_up': 'passed',
            'emergency_wait_s': device.commands[-2]['send_start_s'] - stop_requested,
            'six_axes_per_group': True, 'quantization_unchanged': True}, indent=2))
    finally:
        transport.time = original_time


if __name__ == '__main__':
    main()
