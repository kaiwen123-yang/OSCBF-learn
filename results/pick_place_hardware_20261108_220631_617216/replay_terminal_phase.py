"""Evaluate phase completion on this run's recorded joints; no hardware/simulation."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

out = Path(__file__).resolve().parent
repo = out.parents[1]
sys.path.insert(0, str(repo))
from piper_oscbf import task_reference as current

spec = importlib.util.spec_from_file_location('recorded_task_reference', out/'sources/task_reference.py')
recorded = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recorded
spec.loader.exec_module(recorded)
task = json.loads((out/'task.json').read_text())
controls = list(csv.DictReader((out/'control.csv').open()))
cfg = task['config']
results = []
for step in task['steps']:
    if step['kind'] != 'arm':
        continue
    rows = [r for r in controls if r['stage'] == step['name']]
    if not rows:
        continue
    cap = cfg['stage_velocity_limits_rad_s'].get(step['name'], cfg['joint_velocity_limit_rad_s'])
    oldref, newref = recorded.StageReference(step, cap), current.StageReference(step, cap)
    old = recorded.PhaseGovernor(oldref, cfg['arrival_tolerance_rad'], **cfg.get('reference_governor', {}))
    new = current.PhaseGovernor(newref, cfg['arrival_tolerance_rad'], **cfg.get('reference_governor', {}))
    old_difference, completion = [], None
    eligible = 0
    for row in rows:
        q = [float(row[f'q{i}_rad']) for i in range(1,7)]
        now = float(row['monotonic_s'])
        previous = old.update(q, now=now)
        old_difference.append(abs(previous.phase_s-float(row['phase_phase_s'])))
        if (previous.goal_error_rad <= cfg['arrival_tolerance_rad']
                and oldref.segment(previous.phase_s) == len(oldref.points)-2):
            eligible += 1
        if completion is None:
            revised = new.update(q, now=now)
            if revised.done:
                completion = {'recorded_stage_time_s':float(row['time_s']),
                    'phase':revised.as_dict(), 'recorded_min_h_m':float(row['min_h_m']),
                    'q_measured_rad':q,'emitted_reference_rad':newref.sample(revised.phase_s)[0].tolist(),
                    'all_saved_boundaries_emitted':revised.reference_waypoints_reached == len(newref.points)}
    results.append({'stage':step['name'],'arrival_tolerance_rad':cfg['arrival_tolerance_rad'],
        'last_segment_duration_s':float(newref.times[-1]-newref.times[-2]) if len(newref.times)>1 else 0,
        'last_segment_max_joint_delta_deg':math.degrees(float(abs(newref.points[-1]-newref.points[-2]).max())) if len(newref.points)>1 else 0,
        'old_reconstructed_phase_max_error_s':max(old_difference),
        'original_tolerance_eligible_samples_on_last_segment':eligible,
        'revised_first_completion':completion})
report = {'source':'Recorded encoder trajectory; fixed and archived governors receive identical samples until revised completion. No new dynamics are simulated.',
    'meaning':'Software phase completion on existing samples; not a hardware rerun or physical transfer proof.',
    'stages':results}
(out/'terminal_phase_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results[-1],ensure_ascii=False,indent=2))
