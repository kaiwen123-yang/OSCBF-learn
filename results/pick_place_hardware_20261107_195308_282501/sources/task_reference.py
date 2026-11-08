"""Saved-path references and bounded phase progression for filtered tasks.

This module has no device transport or geometry dependencies. The unfiltered
historical executor keeps its own progression policy; CBF and OSCBF callers use
this common policy and record their actual (possibly different) phase traces.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time

import numpy as np

JOINTS = tuple(f'joint{i}' for i in range(1, 7))


class StageReference:
    """Piecewise-linear reference with the saved executor's time scaling.

    ``cap`` is the effective velocity cap for this stage, after the caller has
    applied the common overall/descent speed scales. Half is reserved for the
    same feed-forward policy as run_pick_place.path_arrays().
    """
    def __init__(self, step, cap):
        self.name = str(step.get('name', 'unnamed'))
        self.cap = float(cap)
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError('Stage velocity cap must be finite and positive')
        trajectory = step['trajectory']['joint_trajectory']
        names = trajectory['joint_names']
        if any(names.count(name) != 1 for name in JOINTS):
            raise ValueError('Saved trajectory must contain each of the six arm joints exactly once')
        order = [names.index(name) for name in JOINTS]
        saved = trajectory['points']
        if not saved:
            raise ValueError('An empty trajectory cannot establish a hold/arrival target')
        self.points = np.asarray([[p['positions'][i] for i in order] for p in saved], dtype=float)
        stamps = np.asarray([p['time_from_start']['sec'] + p['time_from_start']['nanosec']*1e-9
                             for p in saved], dtype=float)
        if self.points.shape != (len(saved), 6) or not np.all(np.isfinite(self.points)):
            raise ValueError('Saved joint positions must be finite six-axis vectors')
        if not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) < 0):
            raise ValueError('Saved trajectory times must be finite and nondecreasing')
        if len(saved) == 1:
            self.times = np.zeros(1)
            self.velocities = np.zeros((0, 6))
        else:
            dq = np.diff(self.points, axis=0)
            durations = np.maximum(np.diff(stamps), np.max(np.abs(dq), axis=1)/(self.cap*.5))
            durations = np.maximum(durations, .01)
            self.times = np.r_[0., np.cumsum(durations)]
            self.velocities = dq / durations[:, None]
        self.duration = float(self.times[-1])
        self.goal = self.points[-1].copy()
        self.is_hold = bool(np.max(np.abs(self.points-self.points[0])) <= 1e-12)
        signature_content = {'schema': 1, 'name': self.name, 'joints': JOINTS,
                             'cap': self.cap, 'times': self.times.tolist(), 'points': self.points.tolist()}
        self.signature = hashlib.sha256(json.dumps(signature_content, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
        for values in (self.points, self.times, self.velocities, self.goal):
            values.setflags(write=False)

    def _phase(self, phase):
        value = float(phase)
        if not math.isfinite(value) or value < 0 or value > self.duration+1e-10:
            raise ValueError('Reference phase is outside this saved stage')
        return min(value, self.duration)

    def segment(self, phase):
        """Right-side segment at a saved waypoint; last segment at the end."""
        phase = self._phase(phase)
        if len(self.points) == 1:
            return None
        return min(len(self.points)-2, max(0, int(np.searchsorted(self.times, phase, side='right')-1)))

    def sample(self, phase):
        phase = self._phase(phase)
        index = self.segment(phase)
        if index is None or phase >= self.duration:
            return self.goal.copy(), np.zeros(6)
        fraction = (phase-self.times[index])/(self.times[index+1]-self.times[index])
        q = self.points[index]*(1-fraction)+self.points[index+1]*fraction
        return q, self.velocities[index].copy()

    def next_boundary(self, phase):
        phase = self._phase(phase)
        index = int(np.searchsorted(self.times, phase, side='right'))
        return float(self.times[index]) if index < len(self.times) else self.duration

    def project_local(self, q, phase):
        """Project onto only the current saved segment, without jumping ahead.

        This is a progress diagnostic. Reference advancement remains separately
        bounded and cannot jump to a globally nearest segment of a looping path.
        """
        q = _joint_vector(q)
        phase = self._phase(phase)
        index = self.segment(phase)
        if index is None:
            return 0.
        vector = self.points[index+1]-self.points[index]
        length_squared = float(vector@vector)
        if length_squared <= 1e-24:
            return phase
        fraction = float(np.clip((q-self.points[index])@vector/length_squared, 0., 1.))
        return float(self.times[index]+fraction*(self.times[index+1]-self.times[index]))

    def tangential_lag(self, q, phase):
        """Return lag in reference seconds and signed along-path error in rad."""
        q = _joint_vector(q)
        desired, derivative = self.sample(phase)
        speed_squared = float(derivative@derivative)
        if speed_squared <= 1e-18:
            return 0., 0.
        numerator = float((desired-q)@derivative)
        return max(0., numerator/speed_squared), numerator/math.sqrt(speed_squared)


def _joint_vector(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError('Measured joints must be a finite six-axis vector')
    return q


@dataclass(frozen=True)
class PhaseState:
    phase_s: float
    rate: float
    requested_rate: float
    actual_phase_rate: float
    elapsed_s: float
    dt_s: float
    tangential_lag_s: float
    tangential_error_rad: float
    projected_phase_s: float
    goal_error_rad: float
    reference_error_rad: float
    no_progress_s: float
    reference_waypoints_reached: int
    paused: bool
    reason: str
    done: bool
    policy: str = 'adaptive_tangential_v1'

    def as_dict(self):
        return asdict(self)


class ReferenceBlockedError(RuntimeError):
    def __init__(self, reason, state):
        self.reason, self.state = reason, state
        super().__init__(f'{reason}: phase={state.phase_s:.6f}s, elapsed={state.elapsed_s:.3f}s, '
                         f'goal_error={state.goal_error_rad:.6f}rad')


class PhaseGovernor:
    """Common finite progression rule for CBF and OSCBF task execution.

    Only positive along-path lag reduces reference speed. A correction transverse
    to the local joint-path tangent does not by itself freeze reference phase.
    A finite no-progress timeout and total-duration limit still reject blocked
    tasks. Completion always requires the caller's unchanged joint arrival
    tolerance; reaching the end of the reference clock alone is insufficient.
    """
    def __init__(self, reference, arrival_tolerance_rad, *, max_tangential_lag_s=.20,
                 no_progress_timeout_s=5., max_duration_s=None, max_phase_step_s=.05,
                 progress_epsilon_rad=1e-4, min_running_rate=.02):
        self.reference = reference
        self.arrival_tolerance_rad = float(arrival_tolerance_rad)
        self.max_tangential_lag_s = float(max_tangential_lag_s)
        self.no_progress_timeout_s = float(no_progress_timeout_s)
        self.max_duration_s = float(4*reference.duration+10 if max_duration_s is None else max_duration_s)
        self.max_phase_step_s = float(max_phase_step_s)
        self.progress_epsilon_rad = float(progress_epsilon_rad)
        self.min_running_rate = float(min_running_rate)
        values = (self.arrival_tolerance_rad, self.max_tangential_lag_s, self.no_progress_timeout_s,
                  self.max_duration_s, self.max_phase_step_s, self.progress_epsilon_rad)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Governor tolerances, times and step bounds must be finite and positive')
        if not 0 < self.min_running_rate < 1:
            raise ValueError('min_running_rate must lie strictly between zero and one')
        self.phase = 0.
        self.started = self.previous = self.last_progress_time = None
        self.last_progress_q = None
        self.best_projection = 0.
        self.best_goal_error = math.inf
        self.last_state = None

    def _rate(self, q, phase):
        if phase >= self.reference.duration:
            return 0.
        lag, _ = self.reference.tangential_lag(q, phase)
        rate = float(np.clip(1.-lag/self.max_tangential_lag_s, 0., 1.))
        return rate if rate >= self.min_running_rate else 0.

    def update(self, q_measured, *, now=None):
        q = _joint_vector(q_measured)
        now = float(time.monotonic() if now is None else now)
        if not math.isfinite(now):
            raise ValueError('Monotonic time must be finite')
        if self.started is None:
            self.started = self.previous = self.last_progress_time = now
            self.last_progress_q = q.copy()
            self.best_goal_error = float(np.max(np.abs(q-self.reference.goal)))
        if now < self.previous:
            raise ValueError('Governor monotonic time moved backwards')
        dt, elapsed = now-self.previous, now-self.started
        previous_phase = self.phase
        requested_rate = self._rate(q, previous_phase)
        requested_step = min(dt, self.max_phase_step_s)*requested_rate
        # Emit a saved boundary before moving into its next segment. Late calls
        # do not skip multiple corners or catch up on an elapsed reference clock.
        self.phase = min(previous_phase+requested_step, self.reference.next_boundary(previous_phase),
                         self.reference.duration)
        self.previous = now
        desired, _ = self.reference.sample(self.phase)
        lag, tangential_error = self.reference.tangential_lag(q, self.phase)
        projection = self.reference.project_local(q, self.phase)
        goal_error = float(np.max(np.abs(q-self.reference.goal)))
        moved = float(np.max(np.abs(q-self.last_progress_q))) > self.progress_epsilon_rad
        forward = projection > self.best_projection+1e-6
        closer = goal_error < self.best_goal_error-self.progress_epsilon_rad
        if moved and (forward or closer):
            self.last_progress_time, self.last_progress_q = now, q.copy()
            self.best_projection = max(self.best_projection, projection)
            self.best_goal_error = min(self.best_goal_error, goal_error)
        at_goal = goal_error <= self.arrival_tolerance_rad
        done = self.phase >= self.reference.duration and at_goal
        waiting_at_hold = self.reference.is_hold and at_goal
        actual_phase_rate = (self.phase-previous_phase)/dt if dt > 0 else 0.
        # Feed-forward uses the observed reference phase rate, including a step
        # shortened at a saved boundary. The first observation has no elapsed
        # interval yet, so it uses the current policy rate.
        rate = (actual_phase_rate if dt > 0 else requested_rate) if self.phase < self.reference.duration else 0.
        if done:
            reason = 'arrived'
        elif self.phase >= self.reference.duration:
            reason = 'awaiting_actual_arrival'
        elif waiting_at_hold:
            reason = 'timed_hold'
        elif rate == 0:
            reason = 'tangential_lag'
        else:
            reason = 'advancing'
        state = PhaseState(phase_s=float(self.phase), rate=rate, requested_rate=requested_rate,
            actual_phase_rate=actual_phase_rate,
            elapsed_s=elapsed, dt_s=dt, tangential_lag_s=lag,
            tangential_error_rad=tangential_error, projected_phase_s=projection,
            goal_error_rad=goal_error, reference_error_rad=float(np.max(np.abs(q-desired))),
            no_progress_s=now-self.last_progress_time,
            reference_waypoints_reached=int(np.searchsorted(self.reference.times, self.phase, side='right')),
            paused=rate == 0 and not done, reason=reason, done=done)
        self.last_state = state
        if not done and elapsed >= self.max_duration_s:
            state = replace(state, rate=0., paused=True, reason='max_duration_exceeded')
            self.last_state = state
            raise ReferenceBlockedError('max_duration_exceeded', state)
        if not done and not waiting_at_hold and state.no_progress_s >= self.no_progress_timeout_s:
            state = replace(state, rate=0., paused=True, reason='no_measured_progress')
            self.last_state = state
            raise ReferenceBlockedError('no_measured_progress', state)
        return state

    def metadata(self):
        return {'schema': 1, 'policy': 'adaptive_tangential_v1', 'stage': self.reference.name,
                'reference_signature': self.reference.signature,
                'lag_metric': 'local_joint_path_tangent; Cartesian transverse displacement is not assumed joint-transverse',
                'reference_duration_s': self.reference.duration,
                'arrival_tolerance_rad': self.arrival_tolerance_rad,
                'max_tangential_lag_s': self.max_tangential_lag_s,
                'no_progress_timeout_s': self.no_progress_timeout_s,
                'max_duration_s': self.max_duration_s, 'max_phase_step_s': self.max_phase_step_s,
                'progress_epsilon_rad': self.progress_epsilon_rad, 'min_running_rate': self.min_running_rate,
                'reference_time_comparison': 'Actual phase traces must be compared. Same policy does not imply a common time schedule.'}


def make_schedule_record(reference, states):
    """Serialize an observed phase trace for later explicit common replay.

    Recording a trace is not replaying it. A future replay consumer must verify
    this reference signature and use these elapsed-time/phase pairs unchanged;
    current adaptive callers must not label their runs a shared-time comparison.
    """
    rows = [v.as_dict() if isinstance(v, PhaseState) else dict(v) for v in states]
    if not rows:
        raise ValueError('An empty phase trace is not a reference schedule')
    elapsed = np.asarray([v['elapsed_s'] for v in rows], dtype=float)
    phase = np.asarray([v['phase_s'] for v in rows], dtype=float)
    policies = {v.get('policy', 'unspecified') for v in rows}
    if (not np.all(np.isfinite(elapsed)) or not np.all(np.isfinite(phase)) or
            elapsed[0] != 0 or phase[0] != 0 or np.any(np.diff(elapsed) <= 0) or
            np.any(np.diff(phase) < 0) or phase[-1] > reference.duration or
            np.any(np.diff(phase) > np.diff(elapsed)+1e-10) or len(policies) != 1):
        raise ValueError('Schedule must start at zero with increasing times and nondecreasing in-range phases')
    payload = {'schema': 1, 'stage': reference.name, 'reference_signature': reference.signature,
               'source_policy': rows[0].get('policy', 'unspecified'),
               'elapsed_s': elapsed.tolist(), 'phase_s': phase.tolist(),
               'record_status': 'observed_trace_not_yet_replayed'}
    payload['schedule_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                             separators=(',', ':')).encode()).hexdigest()
    return payload
