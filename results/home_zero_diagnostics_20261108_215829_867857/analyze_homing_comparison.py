"""Compare the three recorded MOVE_J attempts without connecting to hardware."""
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

output = Path(__file__).resolve().parent
names = ["home_zero_diagnostics_20261108_190550_767544",
         "home_zero_diagnostics_20261108_214707_082496",
         "home_zero_diagnostics_20261108_215829_867857"]
results = []
for name in names:
    root = output.parent / name
    summary = json.loads((root/"summary.json").read_text())
    recording = json.loads((root/"recording.json").read_text())
    raw = list(csv.DictReader((root/"raw_can.csv").open()))
    tx = list(csv.DictReader((root/"sdk_tx.csv").open()))
    feedback = list(csv.DictReader((root/"feedback.csv").open()))
    events = [json.loads(s) for s in (root/"events.jsonl").read_text().splitlines()]
    start = next(e["monotonic_s"] for e in events if e["event"]=="move_j_recording_start")
    errors = [r for r in raw if r["is_error"]=="1"]
    ack = [r for r in errors if int(r["arbitration_id"],16)&0x20]
    bad = [r for r in tx if r["exception"] or r["comm_error_after_send"]]
    zeros = [r for r in feedback if all(float(r[f"q{i}_rad"])==0 for i in range(1,7))]
    gaps = [(float(b["monotonic_s"])-float(a["monotonic_s"]))*1000
            for a,b in zip(feedback,feedback[1:])]
    offsets = [float(r["time_s"])-float(r["send_start_s"]) for r in tx]
    item = {"directory":name,"code_commit":recording["git_commit"],
        "error":summary["error"],"initial_deg":[math.degrees(q) for q in summary["initial_rad"]],
        "last_received_deg":[math.degrees(float(feedback[-1][f"q{i}_rad"])) for i in range(1,7)],
        "final_status":summary.get("final_status"),"final_enabled":summary.get("final_enabled"),
        "first_exact_zero_after_motion_start_s":float(zeros[0]["monotonic_s"])-start if zeros else None,
        "all_samples_zero_after_first_zero":all(all(float(r[f"q{i}_rad"])==0 for i in range(1,7))
            for r in feedback if zeros and float(r["monotonic_s"])>=float(zeros[0]["monotonic_s"])) if zeros else None,
        "raw_frames":len(raw),"error_frames":len(errors),"ack_error_frames":len(ack),
        "error_id_counts":dict(Counter(r["arbitration_id"] for r in errors)),
        "sdk_calls":len(tx),"sdk_error_calls":bad,"complete_feedback_samples":len(feedback),
        "feedback_gap_ms":{"median":statistics.median(gaps),"maximum":max(gaps)},
        "sdk_clock_offset_range_ms":(max(offsets)-min(offsets))*1000,
        "first_error":errors[0] if errors else None,"first_ack":ack[0] if ack else None,
        "first_error_to_enobufs_ms":(float(bad[0]["send_start_s"])-float(errors[0]["receive_monotonic_s"]))*1000 if bad and errors else None,
        "first_ack_to_enobufs_ms":(float(bad[0]["send_start_s"])-float(ack[0]["receive_monotonic_s"]))*1000 if bad and ack else None,
        "interface_delta":recording["interface_delta"],"recorder_error":recording["recorder_error"]}
    results.append(item)
(output/"homing_comparison.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
print(json.dumps([{k:r[k] for k in ("directory","error_frames","ack_error_frames","first_exact_zero_after_motion_start_s","all_samples_zero_after_first_zero","feedback_gap_ms")} for r in results],ensure_ascii=False,indent=2))
