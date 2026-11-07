"""Export the official Piper H+gripper model and inspect geometry/limits offline."""
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

ROOT = Path(__file__).resolve().parents[1]


def main():
    share = Path(get_package_share_directory("agx_arm_description"))
    source = share / "agx_arm_urdf/piper_h/urdf/piper_h_with_gripper_description.xacro"
    result = subprocess.run(["xacro", str(source)], check=True, capture_output=True, text=True)
    robot = ET.fromstring(result.stdout)
    joints = {j.attrib["name"]: j for j in robot.findall("joint")}
    rows = []
    for number in range(1, 7):
        joint = joints[f"joint{number}"]
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"No limits for joint{number}")
        rows.append({"name": joint.attrib["name"], "type": joint.attrib["type"],
                     "parent": joint.find("parent").attrib["link"],
                     "child": joint.find("child").attrib["link"],
                     "axis": joint.find("axis").attrib["xyz"],
                     "limits": {k: float(v) for k, v in limit.attrib.items()}})
    missing = []
    meshes = sorted({node.attrib["filename"] for node in robot.findall(".//mesh")})
    for uri in meshes:
        if uri.startswith("package://"):
            package, relative = uri[len("package://"):].split("/", 1)
            mesh = Path(get_package_share_directory(package)) / relative
        elif uri.startswith("file://"):
            mesh = Path(uri[len("file://"):])
        else:
            mesh = Path(uri)
            if not mesh.is_absolute():
                mesh = source.parent / mesh
        if not mesh.is_file():
            missing.append(str(mesh))
    ROOT.joinpath("models").mkdir(exist_ok=True)
    output = ROOT / "models/piper_h_with_gripper.urdf"
    output.write_text(result.stdout, encoding="utf-8")
    report = {"time": datetime.now().astimezone().isoformat(),
              "urdf": str(output.relative_to(ROOT)), "arm_joints": rows,
              "joint_names": list(joints), "mesh_count": len(meshes),
              "missing_meshes": missing, "tcp_calibrated": False,
              "note": "Official geometry; not yet validated against the physical gripper/TCP."}
    (ROOT / "models/model_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        lim = row["limits"]
        print(row["name"], "axis=", row["axis"], "deg=",
              [round(math.degrees(lim[k]), 4) for k in ("lower", "upper")])
    print("MODEL:", output)
    print("MISSING_MESHES:", missing)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
