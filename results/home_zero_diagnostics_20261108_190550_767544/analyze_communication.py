"""Recalculate this recorded failure; reads files only, never opens CAN."""
import csv
import json
import statistics
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path(__file__).resolve().parent
raw = list(csv.DictReader((root / "raw_can.csv").open()))
tx = list(csv.DictReader((root / "sdk_tx.csv").open()))
summary = json.loads((root / "summary.json").read_text())
recording = json.loads((root / "recording.json").read_text())
errors = [r for r in raw if r["is_error"] == "1"]
ack = [r for r in errors if int(r["arbitration_id"], 16) & 0x20]
normal = [r for r in raw if r["is_error"] == "0" and r["is_rx"] == "1"]
failed = [r for r in tx if r["comm_error_after_send"] or r["exception"]]
origin = float(ack[0]["receive_monotonic_s"])

def point(label, mono, epoch):
    return {"event": label, "monotonic_s": mono,
            "relative_to_first_ack_ms": (mono - origin) * 1000,
            "local_time": datetime.fromtimestamp(epoch, timezone(timedelta(hours=8))).isoformat()}

timeline = [
    point("last_normal_receive", float(normal[-1]["receive_monotonic_s"]), float(normal[-1]["can_timestamp_s"])),
    point("first_ack_error", origin, float(ack[0]["can_timestamp_s"])),
    point("first_enobufs", float(failed[0]["send_start_s"]), float(failed[0]["time_s"])),
    point("application_exception", summary["exception_monotonic_s"], summary["exception_time_s"]),
    point("estop_request_failed", float(tx[-1]["send_start_s"]), float(tx[-1]["time_s"])),
]
offsets = [float(r["time_s"]) - float(r["send_start_s"]) for r in tx]
groups = [float(r["send_start_s"]) for r in tx
          if r["arbitration_id"] == "0x155" and r["phase"] == "home_zero"]
intervals = [(b-a)*1000 for a, b in zip(groups, groups[1:])]
drivers = []
for aid in range(0x261, 0x267):
    samples = [r for r in normal if int(r["arbitration_id"], 16) == aid]
    volts = [int(r["data_hex"][:4], 16) / 10 for r in samples]
    drivers.append({"joint": aid - 0x260, "initial_voltage_V": volts[0],
                    "last_voltage_V": volts[-1], "min_voltage_V": min(volts),
                    "max_voltage_V": max(volts), "last_raw": samples[-1]})
result = {
    "run_commit": recording["git_commit"],
    "run_tracked_source_changes": recording["tracked_source_changes"],
    "raw_frame_count": len(raw), "error_frame_count": len(errors),
    "ack_error_frame_count": len(ack),
    "error_id_counts": dict(Counter(r["arbitration_id"] for r in errors)),
    "sdk_calls": len(tx), "failed_sdk_calls": failed,
    "timeline": timeline,
    "timeline_note": "Relative times use local monotonic callback/call clocks. Receive wall times are SocketCAN timestamps; SDK wall times are call times, not physical bus TX times.",
    "control_150_471_calls": [r for r in tx if r["arbitration_id"] in ("0x150", "0x471")],
    "sdk_clock_offset_range_ms": (max(offsets)-min(offsets))*1000,
    "move_j_group_start_intervals_ms": {"minimum": min(intervals),
        "median": statistics.median(intervals), "maximum": max(intervals),
        "first_interval": intervals[0],
        "remaining_minimum": min(intervals[1:]),
        "remaining_maximum": max(intervals[1:])},
    "driver_voltage_observations": drivers,
    "recorder_error": recording["recorder_error"],
    "interface_delta": recording["interface_delta"],
    "interface_before": json.loads((root/"can_interface_before.json").read_text()),
    "interface_after": json.loads((root/"can_interface_after.json").read_text()),
    "homing_completed": False, "oscbf_retry_started": False,
    "stop_delivery": "0x150/01 request returned ENOBUFS; original summary stop_action=sent is not evidence of delivery",
}
(root / "communication_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({k: result[k] for k in ("error_frame_count", "ack_error_frame_count", "timeline", "sdk_clock_offset_range_ms", "move_j_group_start_intervals_ms")}, ensure_ascii=False, indent=2))
