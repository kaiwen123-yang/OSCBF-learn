"""Recover/analyze saved paired CAN trials without opening a device."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from piper_oscbf.plane import PlaneReference, load_model, make_controller
from scripts.run_piper_plane_pair import analyze
from scripts.run_piper_plane_hardware import save_csv


def read(path):
    with path.open() as stream:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--operator-confirmed-return", action="store_true",
                        help="Use only after the operator confirmed normal completion and return")
    args = parser.parse_args()
    out = args.directory.resolve()
    cfg = json.loads((out/"config.json").read_text())
    rows = {m:read(out/f"{m}.csv") for m in ("oscbf","cbf")}
    reference_rows = read(out/"reference.csv")
    with (out/"feedback.csv").open() as stream:
        feedback = list(csv.DictReader(stream))
    with (out/"commands.csv").open() as stream:
        commands = list(csv.DictReader(stream))
    indexed = {int(r["sequence"]):r for r in feedback}
    report_path = out/"summary.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "status":"completed" if args.operator_confirmed_return else "unconfirmed",
        "error":None,
        "postprocessing_error":"Original JSON export failed on numpy.bool_; recovered from CSV",
        "recovery_basis":"operator confirmed normal motion and return; saved feedback/control/commands",
        "data_source":"measured CAN joint feedback; task pose is model FK",
        "order":["oscbf","cbf"],
        "initial_q_rad":[float(feedback[0][f"q{j}_rad"]) for j in range(1,7)],
        "final_q_rad":[float(feedback[-1][f"q{j}_rad"]) for j in range(1,7)],
    }
    model,_ = load_model()
    reference = PlaneReference(model,cfg)
    report.update(reference_origin_m=reference.p0.tolist(), reference_rotation=reference.rotation.tolist(),
                  plane_bound_m=reference.bound, common_start_target_rad=reference.q0.tolist())
    schedule_time = np.array([r["time_s"] for r in reference_rows])
    for method, data in rows.items():
        rtime = np.array([r["reference_time_s"] for r in data])
        pace = np.array([r["replay_pace"] for r in data])
        selected = [r for r in commands if r["phase"]==method+"_paired"]
        if len(selected) != len(data):
            raise RuntimeError(f"{method}: command/control record counts differ")
        controller = make_controller(model,cfg,method)
        diagnostics=[]
        for row in data:
            q=jnp.array([row[f"q{j}_rad"] for j in range(1,7)])
            value=controller(q,jnp.array([row[f"ref_{a}_m"] for a in "xyz"]),
                             jnp.array([row[f"ref_v_{a}_m_s"] for a in "xyz"]),
                             jnp.asarray(reference.rotation),jnp.asarray(reference.bound))
            jax.block_until_ready(value)
            diag={"time_s":row["time_s"], "sample_sequence":row["sample_sequence"]}
            for j in range(1,7):
                diag[f"u{j}_nom_recomputed_rad_s"]=float(value["u_nom"][j-1])
                diag[f"delta_u{j}_qp_rad_s"]=row[f"u{j}_qp_rad_s"]-float(value["u_nom"][j-1])
                diag[f"delta_u{j}_sdk_rad_s"]=row[f"u{j}_sdk_rad_s"]-float(value["u_nom"][j-1])
            diagnostics.append(diag)
        save_csv(out/f"{method}_corrections.csv",diagnostics)
        h=np.array([r["h_plane_m"] for r in data])
        active=np.array([r["plane_nominal_margin"]<0 for r in data])
        record=report.setdefault(method,{})
        record.update(
            reference_complete=bool(rtime[-1]>=schedule_time[-1]),
            replay_required_extra_slowdown=bool(np.any(pace<1.)),
            extra_slowdown_samples=int(np.sum(pace<1.)),
            first_control_q_rad=[data[0][f"q{j}_rad"] for j in range(1,7)],
            h_negative_samples=int(np.sum(h<0)),
            max_feedback_age_ms=max(r["feedback_age_ms"] for r in data),
            compute_ms_median_p95_max=np.percentile([r["compute_ms"] for r in data],[50,95,100]).tolist(),
            maximum_control_feedback_difference_rad=max(abs(r[f"q{j}_rad"]-
                float(indexed[int(r["sample_sequence"])][f"q{j}_rad"])) for r in data for j in range(1,7)),
            maximum_control_command_difference_rad_s=max(abs(r[f"u{j}_sdk_rad_s"]-
                float(c[f"u{j}_rad_s"])) for r,c in zip(data,selected) for j in range(1,7)),
            reference_position_replay_error_m=max(abs(r[f"ref_{a}_m"]-
                np.interp(r["reference_time_s"],schedule_time,[v[f"ref_{a}_m"] for v in reference_rows]))
                for r in data for a in "xyz"),
            descriptive_active_tangent_rmse_mm=float(np.sqrt(np.mean(
                np.array([r["tangent_error_m"] for r in data])[active]**2))*1000),
            descriptive_active_orientation_rmse_deg=float(np.sqrt(np.mean(
                np.array([r["orientation_error_deg"] for r in data])[active]**2))),
        )
    report["first_control_initial_difference_deg"]=np.rad2deg(
        np.array(report["cbf"]["first_control_q_rad"])-report["oscbf"]["first_control_q_rad"]).tolist()
    report["correction_data_note"]="Nominal control recomputed offline from saved q/reference/config; QP and SDK outputs were recorded during hardware execution."
    analyze(out,rows,report)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (ROOT/"results/plane_pair_latest.txt").write_text(str(out)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
