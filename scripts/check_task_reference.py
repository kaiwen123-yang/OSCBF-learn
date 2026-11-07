"""Bounded, hardware-free checks for the filtered-task reference policy."""
import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_oscbf.task_reference import PhaseGovernor, ReferenceBlockedError, StageReference, make_schedule_record


def step(points, times=None, name='check'):
    times = list(range(len(points))) if times is None else times
    return {'name': name, 'kind': 'arm', 'trajectory': {'joint_trajectory': {
        'joint_names': [f'joint{i}' for i in range(1, 7)],
        'points': [{'positions': list(q), 'time_from_start': {'sec': int(t), 'nanosec': round((t-int(t))*1e9)}}
                   for q, t in zip(points, times)]}}}


def run():
    out = ROOT/'results/task_reference_check_20261107'
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    original = ROOT/'scripts/run_pick_place.py'
    function = next(v for v in ast.parse(original.read_text()).body
                    if isinstance(v, ast.FunctionDef) and v.name == 'path_arrays')
    namespace = {'np': np, 'JOINTS': [f'joint{i}' for i in range(1, 7)]}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(original), 'exec'), namespace)
    counts = 0
    for plan in ['pick_place_plan_20261104_224921_126900', 'pick_place_plan_20261104_232956_473856']:
        task = json.loads((ROOT/'results'/plan/'task.json').read_text())
        for speed, descent in [(1., 1.), (1.5, 1.), (1.5, 1.5)]:
            for s in task['steps']:
                if s['kind'] != 'arm': continue
                cap = task['config']['joint_velocity_limit_rad_s']*(descent if s['name'] in ('approach','lower') else speed)
                reference = StageReference(s, cap)
                times, points = namespace['path_arrays'](s, cap)
                assert np.array_equal(reference.times, times)
                assert np.array_equal(reference.points, points)
                for i, t in enumerate(times):
                    q, qd = reference.sample(t)
                    assert np.allclose(q, points[i], rtol=0, atol=1e-14)
                    expected = ((points[i+1]-points[i])/(times[i+1]-times[i])
                                if i+1 < len(points) else np.zeros(6))
                    assert np.allclose(qd, expected, rtol=0, atol=1e-14)
                counts += 1
    checks['saved_path_timing_and_waypoints'] = {'passed': True, 'stage_scale_combinations': counts}

    zero = np.zeros(6)
    goal = np.array([.20,0,0,0,0,0])
    reference = StageReference(step([zero, goal], [0.,2.]), .30)
    governor = PhaseGovernor(reference, .002, no_progress_timeout_s=2.)
    trace = []
    for tick in range(401):
        q = reference.sample(governor.phase)[0]
        if tick > 20: q[1] = .05
        state = governor.update(q, now=tick*.01)
        trace.append(state)
        if state.phase_s == reference.duration: break
    assert state.phase_s == reference.duration and state.goal_error_rad >= .05-1e-12
    assert not state.done
    assert any(s.reference_error_rad > .01 and s.rate > .5 for s in trace)
    assert all(s.phase_s >= prev.phase_s for prev,s in zip(trace,trace[1:]))
    stop_time = tick*.01
    for n in range(1,52):
        q = goal.copy(); q[1] = .05*(1-n/51)
        state = governor.update(q, now=stop_time+n*.01)
        trace.append(state)
        if state.done: break
    assert state.done and state.goal_error_rad <= .002
    checks['transverse_correction_and_actual_arrival'] = {
        'passed': True, 'transverse_error_rad': .05, 'old_norm_gate_rate_at_this_error': 0.,
        'reference_finished_before_actual_arrival': True, 'arrival_tolerance_rad': .002,
        'actual_done_error_rad': state.goal_error_rad, 'elapsed_s': state.elapsed_s}
    schedule = make_schedule_record(reference, trace)
    (out/'transverse_phase_trace.json').write_text(json.dumps([s.as_dict() for s in trace],indent=2)+'\n')
    (out/'observed_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')

    governor = PhaseGovernor(reference, .002, no_progress_timeout_s=.5)
    caught = None
    for tick in range(101):
        try: governor.update(zero, now=tick*.01)
        except ReferenceBlockedError as exc:
            caught = exc; break
    assert caught and caught.reason == 'no_measured_progress'
    assert not caught.state.done and caught.state.phase_s < .21
    checks['blocked_actual_motion'] = {'passed': True, 'reason': caught.reason, 'state': caught.state.as_dict()}

    governor = PhaseGovernor(reference, .002, no_progress_timeout_s=10., max_duration_s=.4)
    caught = None
    for tick in range(101):
        q = zero.copy(); q[0] = min(.02,tick*.0002)
        try: governor.update(q, now=tick*.01)
        except ReferenceBlockedError as exc:
            caught = exc; break
    assert caught and caught.reason == 'max_duration_exceeded' and not caught.state.done
    checks['finite_total_duration'] = {'passed': True, 'reason': caught.reason, 'elapsed_s': caught.state.elapsed_s}

    one = StageReference(step([goal], [0.]), .30)
    governor = PhaseGovernor(one, .002)
    far = goal.copy(); far[0] += .02
    first = governor.update(far, now=0.)
    assert not first.done and first.phase_s == 0. and first.reason == 'awaiting_actual_arrival'
    near = goal.copy(); near[0] += .001
    final = governor.update(near, now=.5)
    assert final.done and one.duration == 0.
    checks['one_point_hold_requires_arrival'] = {'passed': True, 'initial_error_rad': first.goal_error_rad,
                                                'final_error_rad': final.goal_error_rad}

    timed = StageReference(step([goal,goal], [0.,1.]), .30)
    governor = PhaseGovernor(timed, .002, no_progress_timeout_s=.2)
    for tick in range(101): state = governor.update(goal, now=tick*.01)
    assert state.done and timed.duration == 1.
    checks['timed_stationary_hold'] = {'passed': True, 'duration_s': state.elapsed_s,
                                      'shorter_no_progress_timeout_s': .2}

    points = [zero, np.array([.001,0,0,0,0,0]), np.array([.001,.001,0,0,0,0]), np.array([.002,.001,0,0,0,0])]
    corners = StageReference(step(points,[0.,.01,.02,.03]), .30)
    governor = PhaseGovernor(corners, .002)
    governor.update(points[0], now=0.)
    previous_phase = 0.
    for tick in range(1,4):
        state = governor.update(corners.sample(previous_phase)[0], now=tick*.2)
        assert state.phase_s == corners.times[tick]
        assert state.reference_waypoints_reached == tick+1
        previous_phase = state.phase_s
    checks['late_calls_do_not_skip_waypoints'] = {'passed': True, 'late_dt_s': .2, 'visited_boundaries': 3}

    fast, slow = [], []
    for factor, trace_out in [(1.,fast),(.5,slow)]:
        governor = PhaseGovernor(reference, .002)
        for tick in range(101):
            q = zero.copy(); q[0] = .1*tick*.01*factor
            trace_out.append(governor.update(q,now=tick*.01))
    first_schedule, second_schedule = make_schedule_record(reference,fast), make_schedule_record(reference,slow)
    assert first_schedule['source_policy'] == second_schedule['source_policy']
    assert first_schedule['schedule_sha256'] != second_schedule['schedule_sha256']
    checks['same_policy_not_same_schedule'] = {'passed': True, 'policy': first_schedule['source_policy'],
        'faster_final_phase_s': fast[-1].phase_s, 'slower_final_phase_s': slow[-1].phase_s,
        'schedule_hashes_differ': True}
    report = {'scope':'offline_reference_logic_only_no_hardware_no_geometry', 'status':'passed',
              'checks':checks, 'governor_metadata':PhaseGovernor(reference,.002).metadata(),
              'schedule_replay_status':'Observed traces can be serialized; a fixed-schedule execution consumer is not implemented here.'}
    (out/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': run()
