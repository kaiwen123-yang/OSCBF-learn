"""Replay saved model states; no physics integration and no hardware access."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--method", choices=("cbf", "oscbf"), default="oscbf")
    args = parser.parse_args()
    saved = Path((ROOT / "results/plane_sim_latest.txt").read_text().strip()) if args.result is None else None
    out = args.result or (ROOT / "results" / saved.name)
    cfg = json.loads((out / "config.json").read_text())
    metadata = json.loads((out / "model.json").read_text())
    data = {}
    for method in ("cbf", "oscbf"):
        with (out / f"{method}.csv").open() as file:
            data[method] = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    p.connect(p.GUI)
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        robot = p.loadURDF(str(ROOT / "models/piper_h_fixed_gripper.urdf"), useFixedBase=True)
        joint_map = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}
        joint_indices = [joint_map[name] for name in metadata["joint_order"]]
        p.resetDebugVisualizerCamera(1.05, 45, -22, [0, 0, 0.35])
        first = data[args.method][0]
        p0 = np.array([first[f"p_{a}_m"] for a in "xyz"])
        normal = np.asarray(cfg["plane_normal"])
        plane_center = p0 + normal * cfg["plane_distance_from_start_m"]
        shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.0005, 0.14, 0.14],
                                    rgbaColor=[0.9, 0.3, 0.3, 0.20])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=plane_center,
                          baseOrientation=p.getQuaternionFromEuler([0, 0, float(np.arctan2(normal[1], normal[0]))]))
        colors = {"cbf": [0.81, 0.42, 0.13], "oscbf": [0.09, 0.49, 0.67]}
        for method in data:
            path = [[r[f"p_{a}_m"] for a in "xyz"] for r in data[method][::10]]
            for a, b in zip(path, path[1:]):
                p.addUserDebugLine(a, b, colors[method], lineWidth=3)
        ref_path = [[r[f"ref_{a}_m"] for a in "xyz"] for r in data["cbf"][::10]]
        for a, b in zip(ref_path, ref_path[1:]):
            p.addUserDebugLine(a, b, [0.5, 0.5, 0.5], lineWidth=1)
        marker_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.004, rgbaColor=[0.1, 0.8, 0.2, 1])
        marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_shape)
        print("MODEL REPLAY: C = ordinary CBF, O = OSCBF; Ctrl+C or close window to exit.", flush=True)
        method, index, label = args.method, 0, -1
        while p.isConnected():
            start = time.monotonic()
            keys = p.getKeyboardEvents()
            for key, selected in ((ord("c"), "cbf"), (ord("o"), "oscbf")):
                if keys.get(key, 0) & p.KEY_WAS_TRIGGERED:
                    method, index = selected, 0
            row = data[method][index]
            for i, joint_index in enumerate(joint_indices, 1):
                p.resetJointState(robot, joint_index, row[f"q{i}_rad"])
            p.resetBasePositionAndOrientation(marker, [row[f"ref_{a}_m"] for a in "xyz"], [0, 0, 0, 1])
            if index % 10 == 0:
                text = (f"{method.upper()} | model replay | t={row['time_s']:.1f}s "
                        f"h={row['h_plane_m']*1000:.2f}mm | C/O switch")
                label = p.addUserDebugText(text, [-0.23, 0, 0.82], [0.1, 0.1, 0.1],
                                           textSize=1.2, replaceItemUniqueId=label)
            index = (index + 1) % len(data[method])
            time.sleep(max(0., cfg["dt_s"] - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
