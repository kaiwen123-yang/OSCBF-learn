"""First measured-state OSCBF velocity trial. Default: receive/preview/warmup only."""
import argparse
import csv
import hashlib
import json
import signal
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


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(out, rows):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t = [r["time_s"] for r in rows]
    fields = [("h_plane_m", 1000, "Plane barrier h [mm]"),
              ("tangent_error_m", 1000, "Tangential error [mm]"),
              ("orientation_error_deg", 1, "Orientation error [deg]"),
              ("cycle_period_ms", 1, "Control cycle [ms]")]
    for ax, (field, scale, title) in zip(axes.flat, fields):
        ax.plot(t, [r[field]*scale for r in rows], linewidth=1)
        ax.set(xlabel="Elapsed time [s]", title=title)
        ax.grid(alpha=.3)
    axes[0, 0].axhline(0, color="red", ls=":")
    fig.suptitle("Piper H hardware — measured joint state; end-effector values computed by FK")
    fig.savefig(out / "hardware.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--method", choices=("oscbf", "cbf"), default="oscbf")
    args = parser.parse_args()
    # Installed only in project envs/oscbf. No system/ROS Python changes.
    from piper_oscbf.device import PiperDevice
    cfg = json.loads((ROOT / "config/plane_hardware.json").read_text())
    model, metadata = load_model()
    out = ROOT / "results" / ("plane_hardware_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    report = {"status": "preview", "error": None, "method": args.method,
              "data_source": "CAN joint feedback; flange trajectory is model FK",
              "comparison_status": "single engineering trial, not a matched CBF/OSCBF comparison",
              "task_frame": "flange_link", "reference_frame": "base_link",
              "versions": {name: version(name) for name in
                           ("pyAgxArm", "python-can", "oscbf", "jax", "jaxlib", "qpax", "numpy")},
              "model_sha256": hashlib.sha256(Path(metadata["algorithm_urdf"]).read_bytes()).hexdigest()}
    (out / "model.json").write_text(json.dumps(metadata, indent=2)+"\n")
    device, rows = None, []

    def terminate(*_):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        device = PiperDevice()
        device.connect()
        first, _ = device.state(enabled=False)
        q_initial = np.asarray(device.angles(first))
        q_prepare = q_initial + np.deg2rad(cfg["prepare_delta_deg"])
        if (np.any(q_prepare < np.asarray(model.joint_lower_limits)) or
                np.any(q_prepare > np.asarray(model.joint_upper_limits)) or
                np.any(q_initial < np.asarray(model.joint_lower_limits)) or
                np.any(q_initial > np.asarray(model.joint_upper_limits))):
            raise RuntimeError("Initial or relative target outside official joint bounds")
        cfg["q0_deg"] = np.rad2deg(q_prepare).tolist()
        reference = PlaneReference(model, cfg)
        controller = make_controller(model, cfg, args.method)
        report["initial_q_rad"] = q_initial.tolist()
        report["prepare_target_rad"] = q_prepare.tolist()
        print("当前实测 deg:", np.round(np.rad2deg(q_initial), 3).tolist(), flush=True)
        print("准备目标 deg:", np.round(np.rad2deg(q_prepare), 3).tolist(), "MOVE_J 10%", flush=True)
        print("12 秒闭环：法兰 +X 方向 3 mm 虚拟边界；参考 +X 6 mm、+Z 0.5 mm/s；六轴上限 0.02 rad/s。", flush=True)
        print("正常结束零速后 MOVE_J 10% 返回初始位置并保持使能。Ctrl+C 请求零速及电子急停，可能下沉。", flush=True)
        warm_start = time.perf_counter()
        p, v = reference.at(0.)
        warm = controller(jnp.asarray(q_prepare), jnp.asarray(p), jnp.asarray(v),
                          jnp.asarray(reference.rotation), jnp.asarray(reference.bound))
        jax.block_until_ready(warm)
        report["warmup_s"] = time.perf_counter()-warm_start
        if not bool(warm["converged"]) or not np.isfinite(warm["u"]).all():
            raise RuntimeError("Initial QP warmup did not converge; no motion started")
        print(f"WARMUP_OK: {report['warmup_s']:.3f}s, sigma_min={float(warm['sigma_min']):.6f}", flush=True)
        if not args.execute:
            print("READ_ONLY: 已读取状态并预热算法，未发送使能/运动。", flush=True)
        else:
            device.enable_at(q_initial)
            device.position(q_prepare, "prepare")
            device.robot.set_motion_mode("cpv")
            device.zero()
            anchor, status = device.state()
            if int(status["mode_feedback"]) != 5:
                raise RuntimeError("CPV mode not confirmed")
            cfg["q0_deg"] = np.rad2deg(device.angles(anchor)).tolist()
            reference = PlaneReference(model, cfg)
            rotation, bound = jnp.asarray(reference.rotation), jnp.asarray(reference.bound)
            report["anchor_q_rad"] = device.angles(anchor)
            report["reference_origin_m"] = reference.p0.tolist()
            report["reference_rotation"] = reference.rotation.tolist()
            report["plane_bound_m"] = reference.bound
            device.phase = args.method + "_closed_loop"
            start = previous = deadline = time.monotonic()
            last_print, phase, rate, seq = -1., 0., 1., anchor["sequence"]
            while time.monotonic()-start < cfg["duration_s"]:
                sample, status = device.state(after=seq)
                seq = sample["sequence"]
                now = time.monotonic()
                cycle = now-previous
                phase += cycle*rate
                previous = now
                if int(status["mode_feedback"]) != 5:
                    raise RuntimeError("CPV mode lost during closed-loop control")
                q = jnp.asarray(device.angles(sample))
                desired, derivative = reference.at(phase)
                compute_start = time.perf_counter_ns()
                value = controller(q, jnp.asarray(desired), jnp.asarray(derivative*rate), rotation, bound)
                jax.block_until_ready(value)
                value = {key: np.asarray(x) for key, x in value.items()}
                compute_ms = (time.perf_counter_ns()-compute_start)/1e6
                if not bool(value["converged"]) or not np.isfinite(value["u"]).all():
                    report["failed_qp"] = {"time_s": now-start, "q_rad": np.asarray(q).tolist(),
                                            "converged": bool(value["converged"]),
                                            "iterations": int(value["iterations"])}
                    raise RuntimeError("QP did not converge; velocity trial stopped")
                # Recheck feedback/controller before transmitting after the solve.
                device.state()
                if time.monotonic()-sample["monotonic_s"] > .2:
                    raise RuntimeError("QP input state expired before command transmission")
                wire_u = np.asarray(device.velocity(value["u"]))
                position = value["position"]
                difference = position-desired
                tangent_error = difference-reference.normal*(reference.normal@difference)
                wire_twist = value["jacobian"] @ wire_u
                row = {
                    "time_s": now-start, "sample_sequence": seq, "can_timestamp_s": sample["can_timestamp_s"],
                    "feedback_age_ms": (now-sample["monotonic_s"])*1000,
                    "phase_s": phase, "progress_rate": rate,
                    "h_plane_m": float(value["h_plane"]),
                    "tangent_error_m": float(np.linalg.norm(tangent_error)),
                    "orientation_error_deg": float(np.rad2deg(value["orientation_error_rad"])),
                    "plane_nominal_margin": float(value["plane_nominal_margin"]),
                    "quantized_plane_cbf_margin": float(cfg["alpha_plane"]*value["h_plane"]
                                                       - reference.normal@wire_twist[:3]),
                    "max_qp_violation": float(value["max_qp_violation"]),
                    "sigma_min": float(value["sigma_min"]),
                    "compute_ms": compute_ms, "cycle_period_ms": cycle*1000,
                    "qp_iterations": int(value["iterations"]), "qp_converged": int(value["converged"]),
                }
                for j in range(1, 7):
                    row[f"q{j}_rad"] = float(q[j-1])
                    row[f"u{j}_qp_rad_s"] = float(value["u"][j-1])
                    row[f"u{j}_sdk_rad_s"] = float(wire_u[j-1])
                for k, axis in enumerate("xyz"):
                    row[f"p_{axis}_m"] = float(position[k])
                    row[f"ref_{axis}_m"] = float(desired[k])
                rows.append(row)
                # Uses QP-predicted velocity at this measured state, after SDK
                # rounding. This is not a measured Cartesian velocity estimate.
                rate = float(np.clip(reference.tangent@wire_twist[:3] / cfg["tangent_speed_m_s"], 0, 1))
                if now-start-last_print >= 1:
                    print(f"{now-start:4.1f}s h={row['h_plane_m']*1000:+.3f}mm "
                          f"切向误差={row['tangent_error_m']*1000:.3f}mm "
                          f"姿态误差={row['orientation_error_deg']:.3f}deg "
                          f"max|u|={np.max(np.abs(wire_u)):.3f}rad/s", flush=True)
                    last_print = now-start
                deadline += cfg["dt_s"]
                delay = deadline-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    deadline = time.monotonic()
            device.zero()
            stopped, _ = device.state()
            report["after_zero_q_rad"] = device.angles(stopped)
            returned = device.position(q_initial, "return_initial")
            report["final_q_rad"] = device.angles(returned)
            report["status"] = "completed"
            print("PIPER_PLANE_HARDWARE_FINISHED: 闭环序列完成，已返回初始位置，保持使能。", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if device is not None and device.control_started:
            report["stop_send_errors"] = device.emergency_stop()
        print("ERROR:", report["error"], flush=True)
    finally:
        if device is not None:
            device.disconnect()
            save_csv(out / "feedback.csv", device.feedback)
            save_csv(out / "commands.csv", device.commands)
        save_csv(out / "control.csv", rows)
        if rows:
            field = lambda key: np.array([r[key] for r in rows])
            report.update(samples=len(rows), reference_phase_final_s=float(field("phase_s")[-1]),
                          all_qp_converged=bool(np.all(field("qp_converged"))),
                          minimum_h_mm=float(field("h_plane_m").min()*1000),
                          tangent_rmse_mm=float(np.sqrt(np.mean(field("tangent_error_m")**2))*1000),
                          orientation_rmse_deg=float(np.sqrt(np.mean(field("orientation_error_deg")**2))),
                          max_abs_command_rad_s=max(abs(r[f"u{j}_sdk_rad_s"]) for r in rows for j in range(1, 7)),
                          minimum_sigma=float(field("sigma_min").min()),
                          max_feedback_age_ms=float(field("feedback_age_ms").max()),
                          compute_ms_median_p95_max=np.percentile(field("compute_ms"), [50, 95, 100]).tolist(),
                          cycle_ms_median_p95_max=np.percentile(field("cycle_period_ms")[1:], [50, 95, 100]).tolist()
                          if len(rows) > 1 else None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2)+"\n")
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        plots(out, rows)
        if args.execute:
            (ROOT / "results/plane_hardware_latest.txt").write_text(str(out)+"\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("RESULT_DIR:", out, flush=True)
    return int(report["status"] in ("failed", "interrupted"))


if __name__ == "__main__":
    raise SystemExit(main())
