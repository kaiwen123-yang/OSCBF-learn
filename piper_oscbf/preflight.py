"""Bind a filtered hardware entry to its checked plan, parameters and sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('scripts/run_pick_place.py', 'piper_oscbf/device.py', 'piper_oscbf/pacing.py',
           'piper_oscbf/task_filter.py', 'piper_oscbf/task_geometry.py',
           'piper_oscbf/task_reference.py', 'piper_oscbf/task_session.py',
           'piper_oscbf/task_execution.py', 'piper_oscbf/preflight.py')

def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}

def plan_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_ready(plan, method, speed_scale, descent_speed_scale):
    path = Path(plan).parent/'filter_preflight.json'
    if not path.is_file():
        raise RuntimeError('Filtered task requires scripts/check_filtered_plan.py before hardware')
    proof = json.loads(path.read_text())
    if (proof.get('status') != 'passed' or proof.get('plan_sha256') != plan_hash(plan)
            or proof.get('source_sha256') != source_hashes()
            or method not in proof.get('methods_checked', [])
            or proof.get('speed_scale') != speed_scale
            or proof.get('descent_speed_scale') != descent_speed_scale):
        raise RuntimeError('Filtered preflight does not match current plan/code/parameters; rerun the offline check')
    return proof
