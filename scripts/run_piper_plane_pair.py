"""Hardware record/replay pair: OSCBF records a reference; CBF replays it.

If replay requires extra slowdown, preserve both logs and mark the timing
comparison invalid. No extra slowdown is silently omitted for a nicer result.
"""
import argparse
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
from scripts.run_piper_plane_hardware import save_csv


def progress_rate(cfg, reference, position, desired, nominal_violation):
    """Slow only for positive tangential lag while the QP limits the nominal input.

    Uses measured joint FK and the existing proportional gain. Does not feed
    rounded velocity back into its own feedforward scaling (the old stall).
    Ahead-of-reference motion does not pause a reference trying to catch up.
    """
    if nominal_violation <= 0:
        return 1.
    lag = max(0., float(reference.tangent @ (desired-position)))
    tangent_kp = float(np.dot(reference.tangent**2, np.asarray(cfg["kp_task"][:3])))
    return float(np.clip(1-tangent_kp*lag/cfg["tangent_speed_m_s"], 0, 1))


def run_trial(device, controller, cfg, reference, method, rows, schedule=None, fixed_reference=False):
    device.robot.set_motion_mode("cpv")
    device.zero()
    initial, status = device.state()
    if int(status["mode_feedback"]) != 5:
        raise RuntimeError("CPV mode not confirmed")
    device.phase = method + "_paired"
    rotation, bound = jnp.asarray(reference.rotation), jnp.asarray(reference.bound)
    seq = initial["sequence"]
    start = previous = deadline = time.monotonic()
    phase, rate, replay_time, pace, last_print = 0., 1., 0., 1., -1.
    replay_changed = False
    if schedule is not None:
        times = np.array([r["time_s"] for r in schedule])
        fields = {name: np.array([r[name] for r in schedule]) for name in
                  ("phase_s", "progress_rate", "ref_x_m", "ref_y_m", "ref_z_m",
                   "ref_v_x_m_s", "ref_v_y_m_s", "ref_v_z_m_s")}
    while True:
        now = time.monotonic()
        if schedule is None and now-start >= cfg["duration_s"]:
            break
        if schedule is not None and (replay_time >= times[-1] or now-start >= 2*cfg["duration_s"]):
            break
        sample, status = device.state(after=seq)
        seq = sample["sequence"]
        now = time.monotonic()
        cycle, previous = now-previous, now
        if int(status["mode_feedback"]) != 5:
            raise RuntimeError("CPV mode lost")
        if schedule is None:
            phase += cycle*rate
            desired, derivative = reference.at(phase)
            desired_velocity = derivative*rate
            reference_time = now-start
            applied_rate = rate
        else:
            replay_time = min(times[-1], replay_time+cycle*pace)
            values = {key: float(np.interp(replay_time, times, arr)) for key, arr in fields.items()}
            phase = values["phase_s"]
            desired = np.array([values[f"ref_{a}_m"] for a in "xyz"])
            desired_velocity = np.array([values[f"ref_v_{a}_m_s"] for a in "xyz"])*pace
            reference_time, applied_rate = replay_time, values["progress_rate"]*pace
            replay_changed |= pace < 1.
        q = jnp.asarray(device.angles(sample))
        compute_start = time.perf_counter_ns()
        value = controller(q, jnp.asarray(desired), jnp.asarray(desired_velocity), rotation, bound)
        jax.block_until_ready(value)
        value = {key: np.asarray(v) for key, v in value.items()}
        compute_ms = (time.perf_counter_ns()-compute_start)/1e6
        if not bool(value["converged"]) or not np.isfinite(value["u"]).all():
            raise RuntimeError(f"{method}: QP failed at {now-start:.3f}s, iterations={value['iterations']}")
        device.state()
        if time.monotonic()-sample["monotonic_s"] > .2:
            raise RuntimeError("QP input state expired")
        sent = np.asarray(device.velocity(value["u"]))
        error = value["position"]-desired
        tangent_error = error-reference.normal*(reference.normal@error)
        row = {
            "time_s": now-start, "reference_time_s": reference_time, "phase_s": phase,
            "progress_rate": applied_rate, "replay_pace": pace,
            "sample_sequence": seq, "can_timestamp_s": sample["can_timestamp_s"],
            "feedback_age_ms": (now-sample["monotonic_s"])*1000,
            "h_plane_m": float(value["h_plane"]),
            "tangent_error_m": float(np.linalg.norm(tangent_error)),
            "orientation_error_deg": float(np.rad2deg(value["orientation_error_rad"])),
            "plane_nominal_margin": float(value["plane_nominal_margin"]),
            "nominal_constraint_violation": float(value["nominal_constraint_violation"]),
            "quantized_plane_cbf_margin": float(cfg["alpha_plane"]*value["h_plane"] -
                                                reference.normal@(value["jacobian"][:3]@sent)),
            "max_qp_violation": float(value["max_qp_violation"]),
            "sigma_min": float(value["sigma_min"]), "qp_converged": int(value["converged"]),
            "qp_iterations": int(value["iterations"]), "compute_ms": compute_ms,
            "cycle_period_ms": cycle*1000,
        }
        for j in range(1, 7):
            row[f"q{j}_rad"] = float(q[j-1])
            row[f"u{j}_qp_rad_s"] = float(value["u"][j-1])
            row[f"u{j}_sdk_rad_s"] = float(sent[j-1])
            row[f"u{j}_nom_rad_s"] = float(value["u_nom"][j-1])
            row[f"delta_u{j}_sdk_rad_s"] = float(sent[j-1]-value["u_nom"][j-1])
        for i, a in enumerate("xyz"):
            row[f"p_{a}_m"] = float(value["position"][i])
            row[f"ref_{a}_m"] = float(desired[i])
            row[f"ref_v_{a}_m_s"] = float(desired_velocity[i])
        rows.append(row)
        suggested = progress_rate(cfg, reference, value["position"], desired,
                                  float(value["nominal_constraint_violation"]))
        if fixed_reference:
            rate, pace = 1., 1.
        elif schedule is None:
            rate = suggested
        else:
            recorded_rate = values["progress_rate"]
            pace = min(1., suggested/recorded_rate) if recorded_rate > 0 else 1.
        if now-start-last_print >= 1:
            print(f"{method.upper()} {now-start:4.1f}s ref={reference_time:4.1f}s "
                  f"h={row['h_plane_m']*1000:+.2f}mm 切向误差={row['tangent_error_m']*1000:.2f}mm "
                  f"姿态误差={row['orientation_error_deg']:.2f}deg max|u|={abs(sent).max():.3f}rad/s", flush=True)
            last_print = now-start
        deadline += cfg["dt_s"]
        delay = deadline-time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            deadline = time.monotonic()
    device.zero()
    stopped, _ = device.state()
    return {"initial_q_rad": device.angles(initial), "after_zero_q_rad": device.angles(stopped),
            "reference_complete": bool(schedule is None or replay_time >= times[-1]),
            "replay_required_extra_slowdown": bool(replay_changed)}


def analyze(out, rows, report):
    for method, data in rows.items():
        if not data:
            continue
        field = lambda key: np.array([r[key] for r in data])
        report.setdefault(method, {}).update(
            samples=len(data), all_qp_converged=bool(np.all(field("qp_converged"))),
            minimum_h_mm=float(field("h_plane_m").min()*1000),
            plane_active_samples=int((field("plane_nominal_margin")<0).sum()),
            reference_phase_final_s=float(field("phase_s")[-1]),
            minimum_sigma=float(field("sigma_min").min()),
            max_abs_command_rad_s=max(abs(r[f"u{j}_sdk_rad_s"]) for r in data for j in range(1, 7)),
            fk_displacement_mm=[(data[-1][f"p_{a}_m"]-data[0][f"p_{a}_m"])*1000 for a in "xyz"],
            cycle_ms_median_p95_max=np.percentile(field("cycle_period_ms")[1:], [50,95,100]).tolist()
            if len(data)>1 else None)
    matched = (report["status"] == "completed" and
               report["cbf"].get("reference_complete", False) and
               not report["cbf"].get("replay_required_extra_slowdown", True))
    report["matched_reference_schedule"] = matched
    if matched:
        a, b = rows["oscbf"], rows["cbf"]
        t = np.array([r["time_s"] for r in a])
        bt = np.array([r["reference_time_s"] for r in b])
        active = ((np.array([r["plane_nominal_margin"] for r in a])<0) &
                  (np.interp(t,bt,[r["plane_nominal_margin"] for r in b])<0) &
                  (t>=bt[0]) & (t<=bt[-1]))
        report["common_active_samples"] = int(active.sum())
        for method, data in rows.items():
            times = [r["reference_time_s"] for r in data]
            for key, label, scale in (("tangent_error_m", "active_tangent_rmse_mm", 1000),
                                       ("orientation_error_deg", "active_orientation_rmse_deg", 1)):
                arr = np.interp(t,times,[r[key] for r in data])
                report[method][label] = float(np.sqrt(np.mean(arr[active]**2))*scale) if active.any() else None
    if not any(rows.values()):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method, data in rows.items():
        if not data:
            continue
        t = [r["reference_time_s"] for r in data]
        for ax, key, scale, title in zip(axes.flat,
                ("h_plane_m","tangent_error_m","orientation_error_deg","cycle_period_ms"),
                (1000,1000,1,1), ("Plane h [mm]","Tangential error [mm]","Orientation error [deg]","Control cycle [ms]")):
            ax.plot(t,[r[key]*scale for r in data],label=method.upper(),linewidth=1)
            ax.set(xlabel="Saved reference time [s]",title=title)
            ax.grid(alpha=.3)
            ax.legend()
    axes[0,0].axhline(0,color="red",ls=":")
    label = "matched saved reference" if matched else "timing changed / incomplete: descriptive traces only"
    fig.suptitle("Piper H hardware — "+label+"\nEnd-effector values from measured joints + FK")
    fig.savefig(out/"comparison.png",dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixed-reference", action="store_true",
                        help="Use the same externally prescribed reference timing for both trials; no method-dependent retiming")
    args = parser.parse_args()
    from piper_oscbf.device import PiperDevice
    cfg = json.loads((ROOT/"config/plane_pair_hardware.json").read_text())
    model, metadata = load_model()
    out = ROOT/"results"/("plane_pair_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    report = {"status":"preview", "error":None, "order":["oscbf","cbf"],
              "reference_policy":"fixed_common_timing" if args.fixed_reference else "adaptive_record_replay",
              "data_source":"measured CAN joint feedback; task pose is model FK",
              "versions":{p:version(p) for p in ("pyAgxArm","python-can","oscbf","jax","qpax","numpy")},
              "model_sha256":hashlib.sha256(Path(metadata["algorithm_urdf"]).read_bytes()).hexdigest()}
    (out/"model.json").write_text(json.dumps(metadata,indent=2)+"\n")
    device, rows = None, {"oscbf":[],"cbf":[]}
    def terminate(*_):
        raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM,terminate)
    try:
        device = PiperDevice()
        device.connect()
        first,_ = device.state(enabled=False)
        initial = np.array(device.angles(first))
        prepared = initial+np.deg2rad(cfg["prepare_delta_deg"])
        if (np.any(np.minimum(initial,prepared)<model.joint_lower_limits) or
                np.any(np.maximum(initial,prepared)>model.joint_upper_limits)):
            raise RuntimeError("Initial/prepared position outside official joint bounds")
        cfg["q0_deg"] = np.rad2deg(prepared).tolist()
        reference = PlaneReference(model,cfg)
        controllers = {m:make_controller(model,cfg,m) for m in rows}
        print("当前 deg:",np.round(np.rad2deg(initial),3).tolist(), flush=True)
        print("准备目标 deg:",np.round(np.rad2deg(prepared),3).tolist(),"MOVE_J 10%",flush=True)
        print("边界 +X 15mm；参考 +X 30mm / +Z 8mm/s（12s 最多96mm）；关节上限0.10rad/s。",flush=True)
        print("顺序 OSCBF -> 返回共同起点 -> CBF 重放同一参考 -> 返回本次初始位置；Ctrl+C 零速+电子急停，可能下沉。",flush=True)
        if args.fixed_reference:
            print("固定共同参考时序：两组均不根据各自误差改变参考进度；只用于空旷空间的虚拟边界对比。",flush=True)
        for method, controller in controllers.items():
            p,v=reference.at(0.)
            result=controller(jnp.asarray(prepared),jnp.asarray(p),jnp.asarray(v),
                              jnp.asarray(reference.rotation),jnp.asarray(reference.bound))
            jax.block_until_ready(result)
            if not bool(result["converged"]) or not np.isfinite(result["u"]).all():
                raise RuntimeError(f"{method} warmup failed before motion")
            print("WARMUP_OK:",method,"sigma_min=",float(result["sigma_min"]),flush=True)
        if not args.execute:
            print("READ_ONLY: 已读取/预热，未发使能或运动。",flush=True)
        else:
            device.enable_at(initial)
            device.position(prepared,"prepare_oscbf")
            device.robot.set_motion_mode("cpv")
            device.zero()
            anchor,_=device.state()
            cfg["q0_deg"]=np.rad2deg(device.angles(anchor)).tolist()
            reference=PlaneReference(model,cfg)
            report.update(reference_origin_m=reference.p0.tolist(),reference_rotation=reference.rotation.tolist(),
                          plane_bound_m=reference.bound,initial_q_rad=initial.tolist(),
                          common_start_target_rad=reference.q0.tolist())
            report["oscbf"]=run_trial(device,controllers["oscbf"],cfg,reference,"oscbf",rows["oscbf"],
                                       fixed_reference=args.fixed_reference)
            save_csv(out/"oscbf.csv",rows["oscbf"])
            save_csv(out/"reference.csv",[{k:r[k] for k in
                ("time_s","phase_s","progress_rate","ref_x_m","ref_y_m","ref_z_m",
                 "ref_v_x_m_s","ref_v_y_m_s","ref_v_z_m_s")} for r in rows["oscbf"]])
            report["reference_sha256"]=hashlib.sha256((out/"reference.csv").read_bytes()).hexdigest()
            device.position(reference.q0,"prepare_cbf")
            report["cbf"]=run_trial(device,controllers["cbf"],cfg,reference,"cbf",rows["cbf"],rows["oscbf"],
                                     fixed_reference=args.fixed_reference)
            report["initial_difference_deg"]=np.rad2deg(np.array(report["cbf"]["initial_q_rad"])-
                                                       report["oscbf"]["initial_q_rad"]).tolist()
            returned=device.position(initial,"return_initial")
            report["final_q_rad"]=device.angles(returned)
            report["status"]="completed"
            print("PIPER_PLANE_PAIR_FINISHED: 两轮序列结束，已返回初始位置，保持使能。",flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        report["status"]="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        report["error"]=f"{type(exc).__name__}: {exc}"
        if device is not None and device.control_started:
            report["stop_send_errors"]=device.emergency_stop()
        print("ERROR:",report["error"],flush=True)
    finally:
        if device is not None:
            device.disconnect()
            save_csv(out/"feedback.csv",device.feedback)
            save_csv(out/"commands.csv",device.commands)
        for method,data in rows.items():
            save_csv(out/f"{method}.csv",data)
        (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
        analyze(out,rows,report)
        (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        if args.execute:
            (ROOT/"results/plane_pair_latest.txt").write_text(str(out)+"\n")
        print(json.dumps(report,ensure_ascii=False,indent=2))
        print("RESULT_DIR:",out,flush=True)
    return int(report["status"] in ("failed","interrupted"))


if __name__=="__main__":
    raise SystemExit(main())
