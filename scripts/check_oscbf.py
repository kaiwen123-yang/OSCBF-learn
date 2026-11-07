"""Offline author-model/velocity-QP smoke check. No CAN/ROS or GUI."""
import json
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cbfpy import CBF
from oscbf.core.manipulator import load_panda
from oscbf.examples.dynamic_motion import EESafeSetVelocityConfig


def main():
    robot = load_panda()
    config = EESafeSetVelocityConfig(robot, (0.25, -0.25, 0.25), (0.65, 0.25, 0.65))
    cbf = CBF.from_config(config)
    q = jnp.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    nominal = jnp.zeros(robot.num_joints)
    filt = jax.jit(cbf.safety_filter)
    started = time.perf_counter()
    result = filt(q, nominal).block_until_ready()
    warmup_s = time.perf_counter() - started
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Invalid author-model QP output: {result}")
    durations = []
    for _ in range(10):
        started = time.perf_counter()
        result = filt(q, nominal).block_until_ready()
        durations.append((time.perf_counter() - started) * 1000)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Non-finite output after warmup")
    report = {
        "time": datetime.now().astimezone().isoformat(),
        "robot": "official Franka Panda (not Piper H)",
        "hardware_connected": False,
        "versions": {name: version(name) for name in
                     ("oscbf", "jax", "jaxlib", "cbfpy", "qpax", "numpy", "pybullet")},
        "devices": [str(d) for d in jax.devices()],
        "warmup_s": warmup_s,
        "ten_warm_calls_ms": durations,
        "qdot_output": np.asarray(result).tolist(),
        "note": "Smoke test only, not a control-period benchmark or a Piper H replication result.",
    }
    root = Path(__file__).resolve().parents[1]
    output = root / "logs" / ("oscbf_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("OSCBF_VELOCITY_QP_OK:", output)


if __name__ == "__main__":
    main()
