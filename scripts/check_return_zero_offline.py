"""Offline checks only: no SDK import, CAN socket, or robot object."""
import importlib.util
import json
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("homing", ROOT / "scripts/return_to_zero.py")
homing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(homing)


class HomingChecks(unittest.TestCase):
    def setUp(self):
        audit = json.loads((ROOT / "logs/controller_limits_20261104_203554_580089.json").read_text())
        self.q = [row["q_rad"] for row in audit["joints"]]
        self.bounds = [(row["controller_limits"]["min_angle_limit"],
                        row["controller_limits"]["max_angle_limit"]) for row in audit["joints"]]

    def test_actual_recorded_start_and_all_command_endpoints(self):
        path = homing.zero_path(self.q, self.bounds)
        self.assertEqual(path[0][1:3], [0.0, 0.0])
        self.assertEqual([path[0][i] for i in (0, 3, 4, 5)], [self.q[i] for i in (0, 3, 4, 5)])
        self.assertEqual(path[-1], [0.0]*6)
        for point in path:
            for x, (lo, hi) in zip(point, self.bounds):
                self.assertTrue(lo <= x <= hi)
        for previous, target in zip(path, path[1:]):
            self.assertLessEqual(max(abs(a-b) for a, b in zip(previous, target)), 2*homing.RAD+1e-12)
            self.assertTrue(all(abs(b) <= abs(a) for a, b in zip(previous, target)))

    def test_reject_unreviewed_posture_limits_and_nonfinite(self):
        for index, value in [(0, 12*homing.RAD), (1, -4*homing.RAD),
                             (2, 2*homing.RAD), (5, math.nan)]:
            bad = self.q[:]
            bad[index] = value
            with self.assertRaises(ValueError):
                homing.zero_path(bad, self.bounds)
        bad_bounds = self.bounds[:]
        bad_bounds[4] = (.1, 1.0)
        with self.assertRaises(ValueError):
            homing.zero_path(self.q, bad_bounds)

    def test_complete_fresh_feedback_and_fault_rejection(self):
        q = self.q[:]
        robot = SimpleNamespace(
            get_joint_angles=lambda: SimpleNamespace(msg=q),
            has_comm_error=lambda: False)
        feedback = homing.Feedback(robot)
        for fid in (0x2A5, 0x2A6):
            feedback.observe(SimpleNamespace(is_rx=True, is_remote_frame=False, is_error_frame=False,
                                             arbitration_id=fid, data=bytes(8), timestamp=time.time()))
        self.assertIsNone(feedback.latest)
        feedback.observe(SimpleNamespace(is_rx=True, is_remote_frame=False, is_error_frame=False,
                                         arbitration_id=0x2A7, data=bytes(8), timestamp=time.time()))
        self.assertEqual(feedback.latest[2], q)
        now, wall = time.monotonic(), time.time()
        feedback.drivers = {j: (now, wall, {"foc_status": {"driver_enable_status": False,
                                                         "driver_error_status": False}})
                            for j in range(1, 7)}
        feedback.status = (now, wall, {"arm_status": 0, "err_status": {"fault": False}})
        self.assertEqual(feedback.read(required_enabled=False)[0], q)
        with self.assertRaises(RuntimeError):
            feedback.read(required_enabled=True)
        feedback.drivers[3][2]["foc_status"]["driver_error_status"] = True
        with self.assertRaises(RuntimeError):
            feedback.read()
        feedback.drivers[3][2]["foc_status"]["driver_error_status"] = False
        feedback.latest = (now-1, wall-1, q)
        with self.assertRaises(RuntimeError):
            feedback.read()


if __name__ == "__main__":
    unittest.main(verbosity=2)
