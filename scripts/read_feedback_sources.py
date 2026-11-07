"""Receive only: compare joint, motor and flange feedback without changing mode."""
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    props = subprocess.check_output(
        ["udevadm", "info", "--query=property", "--path=/sys/class/net/can0"], text=True)
    if "ID_SERIAL_SHORT=004F00374548570120353731" not in props.splitlines():
        raise RuntimeError("USB-CAN identity mismatch")
    robot = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.PIPER_H, firmeware_version=PiperFW.V189,
        channel="can0", interface="socketcan", bitrate=1000000,
        auto_connect=False, enable_check_can=True, timeout=.1))
    counter = Path("/sys/class/net/can0/statistics/tx_packets")
    seen = set()
    try:
        before = int(counter.read_text())
        robot.connect(start_read_thread=False)
        comm = robot.get_context().get_comm()
        start = time.monotonic()
        while time.monotonic()-start < 3:
            frame = comm.recv()
            if frame is not None and frame.is_rx and not frame.is_error_frame:
                seen.add(frame.arbitration_id)
        required = {0x2A1, *range(0x2A2, 0x2A8), *range(0x251, 0x257)}
        if not required <= seen:
            raise RuntimeError(f"Missing feedback frames: {sorted(required-seen)}")
        joint = robot.get_joint_angles()
        motors = [robot.get_motor_states(j) for j in range(1, 7)]
        flange = robot.get_flange_pose()
        q = list(joint.msg)
        qm = [m.msg.position for m in motors]
        report = {
            "time": datetime.now().astimezone().isoformat(),
            "joint_feedback_rad": q,
            "motor_position_rad": qm,
            "joint_feedback_deg": [math.degrees(x) for x in q],
            "motor_position_deg": [math.degrees(x) for x in qm],
            "motor_minus_joint_deg": [math.degrees(a-b) for a, b in zip(qm, q)],
            "motor_velocity_rad_s": [m.msg.velocity for m in motors],
            "joint_timestamp_s": joint.timestamp,
            "motor_timestamps_s": [m.timestamp for m in motors],
            "controller_flange_pose_m_rad": list(flange.msg),
            "flange_timestamp_s": flange.timestamp,
            "arm_status": robot.get_arm_status().msg.to_dict(),
            "interface_tx_delta": int(counter.read_text())-before,
            "note": "Passive CAN snapshot. Motor and joint feedback are not assumed equivalent.",
        }
    finally:
        robot.disconnect()
    output = ROOT / "logs" / ("feedback_sources_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".json")
    text = json.dumps(report, ensure_ascii=False, indent=2)+"\n"
    output.write_text(text)
    (ROOT / "logs/feedback_sources_latest.json").write_text(text)
    print(text)
    print("SAVED:", output)


if __name__ == "__main__":
    main()
