"""Run both velocity filters on the same ideal Piper H kinematic plant/reference."""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from piper_oscbf.plane import PlaneReference, load_model, make_controller

METHODS = ("cbf", "oscbf")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_results(out, rows, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    colors = {"cbf": "#cf6b22", "oscbf": "#167eaa"}
    for method in METHODS:
        data = rows[method]
        if not data:
            continue
        series = lambda name: np.array([r[name] for r in data])
        t = series("time_s")
        kw = {"label": method.upper(), "color": colors[method]}
        ax[0, 0].plot(series("tangent_position_m")*1000, series("normal_position_m")*1000, **kw)
        ax[0, 1].plot(t, series("tangent_error_m")*1000, **kw)
        ax[1, 0].plot(t, series("orientation_error_deg"), **kw)
        ax[1, 1].plot(t, series("h_plane_m")*1000, **kw)
        ax[2, 0].plot(t, series("compute_ms"), linewidth=0.6, **kw)
    first = rows["cbf"]
    if first:
        pref = np.array([[r[f"ref_{k}_m"] for k in "xyz"] for r in first]) - ref.p0
        ax[0, 0].plot(pref @ ref.tangent * 1000, pref @ ref.normal * 1000, "k--", label="Reference")
        ax[0, 0].axhline(ref.cfg["plane_distance_from_start_m"]*1000, color="red", ls=":", label="Plane")
        ax[2, 1].plot([r["time_s"] for r in first], [r["phase_s"] for r in first], label="Shared phase")
    labels = [
        ("Model FK path", "Tangential position [mm]", "Normal position [mm]"),
        ("Tangential tracking error", "Simulation time [s]", "Error [mm]"),
        ("Orientation error", "Simulation time [s]", "Geodesic error [deg]"),
        ("Plane barrier (h >= 0)", "Simulation time [s]", "h [mm]"),
        ("Warmed control computation", "Simulation time [s]", "Compute time [ms]"),
        ("Common reference progress", "Simulation time [s]", "Phase [s]"),
    ]
    for axis, (title, x, y) in zip(ax.flat, labels):
        axis.set(title=title, xlabel=x, ylabel=y)
        axis.grid(alpha=0.25)
        axis.legend()
    ax[1, 1].axhline(0, color="red", ls=":")
    fig.suptitle("Piper H / flange_link — ideal velocity simulation, not hardware measurements")
    fig.savefig(out / "comparison.png", dpi=170)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/plane_sim.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    robot, metadata = load_model()
    ref = PlaneReference(robot, cfg)
    out = ROOT / "results" / ("plane_sim_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    controls = {m: make_controller(robot, cfg, m) for m in METHODS}
    states = {m: ref.q0.copy() for m in METHODS}
    rows = {m: [] for m in METHODS}
    warmup = {}
    p, v = ref.at(0.)
    print("SIMULATION_ONLY: base_link -> flange_link; no CAN or SDK access.", flush=True)
    for method in METHODS:
        start = time.perf_counter()
        value = controls[method](jnp.asarray(ref.q0), jnp.asarray(p), jnp.asarray(v))
        jax.block_until_ready(value)
        warmup[method] = time.perf_counter() - start
        print(f"WARMUP {method}: {warmup[method]:.3f} s", flush=True)
    dt, phase, progress_rate = cfg["dt_s"], 0., 1.
    status, error = "completed", None
    try:
        for k in range(round(cfg["duration_s"] / dt)):
            desired, derivative = ref.at(phase)
            values = {}
            candidates = []
            for method in METHODS:
                q = states[method]
                start = time.perf_counter_ns()
                result = controls[method](jnp.asarray(q), jnp.asarray(desired),
                                          jnp.asarray(derivative * progress_rate))
                jax.block_until_ready(result)
                compute_ms = (time.perf_counter_ns() - start) / 1e6
                value = {name: np.asarray(x) for name, x in result.items()}
                values[method] = value
                difference = value["position"] - desired
                tangent_error = difference - ref.normal * (ref.normal @ difference)
                row = {
                    "time_s": k*dt, "phase_s": phase, "progress_rate": progress_rate,
                    "h_plane_m": float(value["h_plane"]),
                    "tangent_error_m": float(np.linalg.norm(tangent_error)),
                    "orientation_error_deg": float(np.rad2deg(value["orientation_error_rad"])),
                    "normal_position_m": float(ref.normal @ (value["position"]-ref.p0)),
                    "tangent_position_m": float(ref.tangent @ (value["position"]-ref.p0)),
                    "plane_nominal_margin": float(value["plane_nominal_margin"]),
                    "plane_applied_margin": float(value["plane_applied_margin"]),
                    "max_qp_violation": float(value["max_qp_violation"]),
                    "min_joint_margin_rad": float(value["min_joint_margin_rad"]),
                    "sigma_min": float(value["sigma_min"]),
                    "compute_ms": compute_ms, "converged": int(value["converged"]),
                    "iterations": int(value["iterations"]),
                }
                for j in range(6):
                    row[f"q{j+1}_rad"] = float(q[j])
                    row[f"u{j+1}_rad_s"] = float(value["u"][j])
                    row[f"u_nom{j+1}_rad_s"] = float(value["u_nom"][j])
                for j, axis in enumerate("xyz"):
                    row[f"p_{axis}_m"] = float(value["position"][j])
                    row[f"ref_{axis}_m"] = float(desired[j])
                    row[f"ref_v_{axis}_m_s"] = float(derivative[j]*progress_rate)
                rows[method].append(row)
                if not bool(value["converged"]) or not np.isfinite(value["u"]).all():
                    raise RuntimeError(f"{method}: QP did not converge at t={k*dt:.3f}s")
                # One common phase for both experiments. The deliberately blocked
                # normal component is not counted as requested tangential progress.
                candidates.append(float(np.clip(ref.tangent @ value["task_velocity"][:3] /
                                                cfg["tangent_speed_m_s"], 0., 1.)))
            for method in METHODS:
                states[method] += dt * values[method]["u"]
            phase += dt * progress_rate
            progress_rate = min(candidates)
    except (Exception, KeyboardInterrupt) as exc:
        status, error = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", repr(exc)
    for method in METHODS:
        write_csv(out / f"{method}.csv", rows[method])
    # Compare identical indices for which BOTH nominal commands violate the plane CBF.
    count = min(len(rows[m]) for m in METHODS)
    common_active = np.array([all(rows[m][k]["plane_nominal_margin"] < 0 for m in METHODS)
                              for k in range(count)], dtype=bool)
    summary = {
        "status": status, "error": error, "data_source": "ideal_velocity_simulation_model_FK",
        "task_frame": metadata["task_frame"], "completed_time_s": count*dt,
        "shared_phase_final_s": phase, "common_active_samples": int(common_active.sum()),
        "warmup_seconds": warmup,
        "versions": {p: version(p) for p in ("oscbf", "jax", "jaxlib", "qpax", "numpy")},
        "model_sha256": hashlib.sha256(Path(metadata["algorithm_urdf"]).read_bytes()).hexdigest(),
        "timing_scope": "FK + nominal + QP + diagnostics + JAX synchronization; excludes logging and any device IO",
    }
    for method in METHODS:
        data = rows[method][:count]
        if not data:
            continue
        field = lambda name: np.array([r[name] for r in data])
        final_p = np.asarray(robot.ee_position(jnp.asarray(states[method])))
        summary[method] = {
            "samples": len(data), "all_qp_converged": bool(np.all(field("converged"))),
            "minimum_h_m": min(float(field("h_plane_m").min()), float(ref.bound-ref.normal@final_p)),
            "max_qp_violation": float(field("max_qp_violation").max()),
            "max_abs_command_rad_s": max(abs(r[f"u{j}_rad_s"]) for r in data for j in range(1, 7)),
            "minimum_sigma": float(field("sigma_min").min()),
            "min_joint_margin_rad": float(field("min_joint_margin_rad").min()),
            "compute_ms_median_p95_max": np.percentile(field("compute_ms"), [50, 95, 100]).tolist(),
            "active_tangent_rmse_mm": (float(np.sqrt(np.mean(field("tangent_error_m")[common_active]**2))*1000)
                                       if common_active.any() else None),
            "active_orientation_rmse_deg": (float(np.sqrt(np.mean(field("orientation_error_deg")[common_active]**2)))
                                            if common_active.any() else None),
            "final_q_rad": states[method].tolist(),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(out, rows, ref)
    (ROOT / "results/plane_sim_latest.txt").write_text(str(out) + "\n")
    print(json.dumps(summary, indent=2))
    print("RESULT_DIR:", out)
    print("PIPER_PLANE_SIM_" + status.upper())
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
