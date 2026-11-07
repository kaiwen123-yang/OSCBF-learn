"""Derive a six-DOF OSCBF model from the already-exported official ROS URDF.

Offline only. Gripper joints are fixed at nominal zero opening. This first
model uses flange_link, not an invented or calibrated grasp TCP.
"""
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from oscbf.core.manipulator import Manipulator
from oscbf.utils.urdf_parser import parse_urdf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ARM_JOINTS = [f"joint{j}" for j in range(1, 7)]


def origin_transform(joint):
    origin = joint.find("origin")
    transform = np.eye(4)
    if origin is not None:
        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def main():
    source = MODELS / "piper_h_with_gripper.urdf"
    tree = ET.parse(source)
    root = tree.getroot()
    fixed = []
    for joint in root.findall("joint"):
        if joint.get("name") not in ARM_JOINTS and joint.get("type") != "fixed":
            fixed.append(joint.get("name"))
            joint.set("type", "fixed")
            for tag in ("mimic", "limit", "axis", "dynamics"):
                for child in joint.findall(tag):
                    joint.remove(child)
    flange_joint = next(j for j in root.findall("joint") if j.get("name") == "flange_joint")
    if flange_joint.find("parent").get("link") != "link6":
        raise RuntimeError("Expected official link6 -> flange_link attachment")
    ee_offset = origin_transform(flange_joint)
    # Preserve a portable geometric copy. STL avoids optional COLLADA dependencies.
    portable = copy.deepcopy(root)
    share = ROOT / "ros_ws/src/agx_arm_ros/src/agx_arm_description"
    for mesh in portable.findall(".//mesh"):
        uri = mesh.get("filename")
        prefix = "package://agx_arm_description/"
        if not uri.startswith(prefix):
            raise RuntimeError(f"Unrecognized mesh URI: {uri}")
        path = share / uri[len(prefix):]
        if path.suffix.lower() == ".dae":
            path = path.parent.parent / (path.stem+".stl")
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", os.path.relpath(path, MODELS))
    portable_path = MODELS / "piper_h_fixed_gripper.urdf"
    ET.indent(portable)
    ET.ElementTree(portable).write(portable_path, encoding="utf-8", xml_declaration=True)
    # The author parser needs the chain and inertias, not mesh loading.
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for child in link.findall(tag):
                link.remove(child)
    algorithm_path = MODELS / "piper_h_oscbf.urdf"
    ET.indent(root)
    ET.ElementTree(root).write(algorithm_path, encoding="utf-8", xml_declaration=True)
    parsed = parse_urdf(str(algorithm_path))
    if parsed["joint_names"] != ARM_JOINTS:
        raise RuntimeError(f"Unexpected joint order: {parsed['joint_names']}")
    robot = Manipulator.from_urdf(str(algorithm_path), ee_offset=ee_offset)
    feedback_path = ROOT / "logs/feedback_sources_latest.json"
    feedback = json.loads(feedback_path.read_text()) if feedback_path.is_file() else None
    configurations = {"zero": np.zeros(6),
                      "bent_simulation_only": np.deg2rad([0, 30, -60, 0, 30, 0])}
    if feedback is not None:
        configurations["recorded_joint_feedback"] = np.asarray(feedback["joint_feedback_rad"])
        configurations["recorded_motor_position"] = np.asarray(feedback["motor_position_rad"])
    results = {}
    for name, q in configurations.items():
        transform = np.asarray(robot.ee_transform(jnp.asarray(q)))
        jacobian = np.asarray(robot.ee_jacobian(jnp.asarray(q)))
        result = {"q_rad": q.tolist(), "flange_transform": transform.tolist(),
                  "flange_xyz_m": transform[:3, 3].tolist(),
                  "flange_rpy_rad": Rotation.from_matrix(transform[:3, :3]).as_euler("xyz").tolist(),
                  "jacobian_shape": list(jacobian.shape),
                  "jacobian_singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist()}
        if name.startswith("recorded_"):
            pose = np.asarray(feedback["controller_flange_pose_m_rad"])
            rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
            result["position_difference_from_controller_mm"] = float(np.linalg.norm(transform[:3, 3]-pose[:3])*1000)
            result["rotation_difference_from_controller_deg"] = float(np.degrees(
                Rotation.from_matrix(rotation.T @ transform[:3, :3]).magnitude()))
        results[name] = result
        print(name, "q_deg=", np.round(np.rad2deg(q), 3).tolist(),
              "flange_xyz_m=", np.round(transform[:3, 3], 6).tolist(),
              "J_shape=", jacobian.shape, flush=True)
        print("  singular_values:", result["jacobian_singular_values"], flush=True)
        if name.startswith("recorded_"):
            print("  difference_from_controller:", result["position_difference_from_controller_mm"], "mm;",
                  result["rotation_difference_from_controller_deg"], "deg", flush=True)
    report = {
        "time": datetime.now().astimezone().isoformat(),
        "source_urdf": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "algorithm_urdf": str(algorithm_path.relative_to(ROOT)), "portable_urdf": str(portable_path.relative_to(ROOT)),
        "robot": "piper_h", "num_joints": robot.num_joints, "joint_order": parsed["joint_names"],
        "reference_frame": "base_link", "task_frame": "flange_link", "ee_offset": ee_offset.tolist(),
        "tcp_calibrated": False, "gripper_fixed_joints": fixed, "gripper_nominal_opening_m": 0.0,
        "cbf_collision_spheres_defined": False,
        "feedback_source": str(feedback_path.relative_to(ROOT)) if feedback else None,
        "feedback_time": feedback["time"] if feedback else None,
        "configurations": results,
        "notes": ["All FK positions are model calculations, not camera/metrology measurements.",
                  "Bent configuration is for offline evaluation only; no hardware motion is sent.",
                  "Preserved official limits/inertias; robot.joint_max_velocities is not an approved hardware speed cap.",
                  "Model loaded successfully does not establish a collision-safe controller or calibrated grasp TCP."],
    }
    output = MODELS / "piper_h_oscbf_model.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print("PIPER_H_OSCBF_MODEL_READY: DOF=", robot.num_joints, "task_frame=flange_link", flush=True)
    print("MODEL_REPORT:", output)


if __name__ == "__main__":
    main()
