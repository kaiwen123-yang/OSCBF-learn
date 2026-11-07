"""IROS 2025 OSCBF, IV-B (18)-(28), specialized to full-rank six-axis control.

Uses the authors' Manipulator and nominal controller and their QP solver.
No device transport, mode switching, or hardware commands are imported here.
"""
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import qpax
from oscbf.core.controllers import PoseTaskVelocityController
from oscbf.core.manipulator import Manipulator

ROOT = Path(__file__).resolve().parents[1]


def load_model():
    metadata = json.loads((ROOT / "models/piper_h_oscbf_model.json").read_text())
    # Use this checkout's model; archived reports retain their original paths.
    metadata["algorithm_urdf"] = str(ROOT / "models/piper_h_oscbf.urdf")
    metadata["portable_urdf"] = str(ROOT / "models/piper_h_fixed_gripper.urdf")
    robot = Manipulator.from_urdf(metadata["algorithm_urdf"],
                                 ee_offset=np.asarray(metadata["ee_offset"]))
    return robot, metadata


class PlaneReference:
    def __init__(self, robot, cfg):
        self.q0 = np.deg2rad(cfg["q0_deg"])
        initial = np.asarray(robot.ee_transform(jnp.asarray(self.q0)))
        self.p0, self.rotation = initial[:3, 3], initial[:3, :3]
        self.normal = np.asarray(cfg["plane_normal"])
        self.tangent = np.asarray(cfg["tangent_direction"])
        self.bound = float(self.normal @ self.p0 + cfg["plane_distance_from_start_m"])
        self.cfg = cfg

    def at(self, phase):
        c = self.cfg
        a = np.clip(phase / c["normal_ramp_s"], 0.0, 1.0)
        normal_distance = c["normal_reference_travel_m"] * (3*a*a - 2*a*a*a)
        normal_speed = c["normal_reference_travel_m"] * 6*a*(1-a) / c["normal_ramp_s"]
        position = (self.p0 + self.normal * normal_distance +
                    self.tangent * c["tangent_speed_m_s"] * phase)
        derivative = self.normal * normal_speed + self.tangent * c["tangent_speed_m_s"]
        return position, derivative


def make_controller(robot, cfg, method):
    """Return one jitted control step consuming the current joint state.

    Simulation passes integrated state. The hardware entry passes a new measured
    joint state each cycle and supplies its measured reference anchor dynamically.
    """
    reference = PlaneReference(robot, cfg)
    normal = jnp.asarray(reference.normal)
    lower, upper = jnp.asarray(robot.joint_lower_limits), jnp.asarray(robot.joint_upper_limits)
    eye = jnp.eye(6)
    weights_squared = jnp.asarray(cfg["task_weights"]) ** 2
    nominal = PoseTaskVelocityController(6, jnp.asarray(cfg["kp_task"]),
                                         jnp.zeros(6), None, None)
    reference_rotation = jnp.asarray(reference.rotation)

    @jax.jit
    def step(q, desired_position, desired_velocity, desired_rotation=None, plane_bound=None):
        # Hardware passes its measured anchor after positioning, without JIT recompilation.
        desired_rotation = reference_rotation if desired_rotation is None else desired_rotation
        plane_bound = reference.bound if plane_bound is None else plane_bound
        transform = robot.ee_transform(q)
        position, rotation = transform[:3, 3], transform[:3, :3]
        jacobian = robot.ee_jacobian(q)
        u_nom = nominal(q, position, rotation, desired_position, desired_rotation,
                        desired_velocity, jnp.zeros(3), q, jacobian)
        # For full-rank 6x6 J, N=0 in paper Eq. (25). No nullspace task.
        metric = (jacobian.T @ (weights_squared[:, None] * jacobian)
                  if method == "oscbf" else eye)
        metric = (metric + metric.T) / 2 * cfg.get("qp_objective_scale", 1.0)
        h_plane = plane_bound - normal @ position
        # SDK CPV rounds velocity to 0.001 rad/s. Account for bounded rounding
        # error in the linear inequalities; disabled for the original simulation.
        half_quantum = cfg.get("velocity_quantization_rad_s", 0.0) / 2
        # G u <= b: -hdot <= alpha*h; explicit joint CBFs and speed limits.
        G = jnp.concatenate(((normal @ jacobian[:3])[None, :], -eye, eye, eye, -eye))
        b = jnp.concatenate((jnp.array([cfg["alpha_plane"] * h_plane
                                        - half_quantum*jnp.sum(jnp.abs(G[0]))]),
                             cfg["alpha_joint"] * (q - lower) - half_quantum,
                             cfg["alpha_joint"] * (upper - q) - half_quantum,
                             jnp.full(12, cfg["velocity_limit_rad_s"] - half_quantum)))
        u, _, _, _, converged, iterations = qpax.solve_qp(
            metric, -metric @ u_nom, jnp.zeros((0, 6)), jnp.zeros(0), G, b,
            solver_tol=cfg["solver_tol"], max_iter=cfg["solver_max_iter"])
        task_velocity = jacobian @ u
        orientation_error = jnp.arccos(jnp.clip(
            (jnp.trace(desired_rotation.T @ rotation) - 1) / 2, -1., 1.))
        return {
            "u": u, "u_nom": u_nom, "position": position, "jacobian": jacobian,
            "task_velocity": task_velocity, "h_plane": h_plane,
            "orientation_error_rad": orientation_error,
            "plane_nominal_margin": b[0] - G[0] @ u_nom,
            "plane_applied_margin": b[0] - G[0] @ u,
            "max_qp_violation": jnp.maximum(0., jnp.max(G @ u - b)),
            "nominal_constraint_violation": jnp.max(G @ u_nom - b),
            "min_joint_margin_rad": jnp.min(jnp.concatenate((q-lower, upper-q))),
            "sigma_min": jnp.linalg.svd(jacobian, compute_uv=False)[-1],
            "converged": converged, "iterations": iterations,
        }
    return step
