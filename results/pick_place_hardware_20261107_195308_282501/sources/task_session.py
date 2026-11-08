"""Task filter lifecycle and aligned recording, without device or timing calls.

The executor owns CAN, scheduling, actual arrival and physical gripper actions.
This session starts with a world cube at the configured pick position and an
empty gripper. Model attachment is not evidence that the physical grasp worked.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from piper_oscbf.task_filter import TaskFilterError, TaskSafetyFilter
from piper_oscbf.task_reference import PhaseGovernor, StageReference, make_schedule_record

ROOT = Path(__file__).resolve().parents[1]
ARRAY_FIELDS = ('h', 'physical_h', 'nominal_margin', 'qp_margin', 'applied_margin',
                'robust_qp_margin', 'constraint_mask', 'u_nom', 'u_qp', 'u_sdk')


def jsonable(value):
    """Preserve failure diagnostics in strict JSON, including nonfinite numbers."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'as_dict'):
        return jsonable(value.as_dict())
    return repr(value)


def _canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path):
    path = ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class TaskSession:
    def __init__(self, task, method):
        self.task = copy.deepcopy(task)
        self.cfg = self.task['config']
        self.method = method
        self.filter = TaskSafetyFilter(self.cfg, method)
        self.geometry = self.filter.geometry
        self.carrying = False
        self.world_cube_T = np.asarray(self.geometry.world_cube_transform(), float)
        self.payload_T_tcp = np.asarray(self.geometry.default_payload_T_tcp, float).copy()
        self.events = []
        self.failures = []
        self.control = []
        self.holds = []
        self.schedules = {}
        self._finished = None
        self.geometry_constraint_count = self.filter.geometry_constraint_count
        self.constraint_labels = tuple(self.filter.constraint_labels)
        self.h_labels = self.constraint_labels
        self.governor_options = copy.deepcopy(self.cfg.get('reference_governor', {}))
        self._fingerprint_payloads = self._common_payloads()
        self.fingerprint = {name: _digest(value) for name, value in self._fingerprint_payloads.items()}
        self.warmup = self.filter.warmup(np.asarray(self.cfg['initial_q_rad'], float),
                                        cap=float(self.cfg['joint_velocity_limit_rad_s']))
        # A nonmoving initial hold also verifies the actual initial scene, unlike
        # the filter's deliberately scene-independent solver compilation probe.
        initial = self.validate_hold(np.asarray(self.cfg['initial_q_rad'], float),
            float(self.cfg['gripper_open_m']), float(self.cfg['joint_velocity_limit_rad_s']), 'open')
        self.initial_hold = {key: jsonable(initial[key]) for key in
            ('solver_status', 'sigma_min', 'min_h_m', 'min_physical_distance_m', 'u_sdk')}
        self.events.append({'event': 'session_initialized', 'event_index': 0,
            'carrying': False, 'world_cube_T': self.world_cube_T.tolist(),
            'object_state_source': 'configured scene, not a perception measurement'})

    def _stage_cap(self, stage):
        return float(self.cfg.get('stage_velocity_limits_rad_s', {}).get(
            stage, self.cfg['joint_velocity_limit_rad_s']))

    def _common_payloads(self):
        f = self.filter
        geometry_metadata = self.geometry.metadata()
        physical_keys = ('frame', 'scene_status', 'tcp_offset_m', 'tcp_rpy_rad', 'tcp_status',
            'cube_size_m', 'pick_xy_m', 'place_xy_m', 'table_top_z_m', 'table_center_xy_m',
            'table_size_xy_m', 'tip_clearance_m', 'obstacles', 'held_cube_tcp_offset_m',
            'held_cube_orientation_assumption', 'field_verification_required')
        governors = {}
        for step in self.task['steps']:
            if step['kind'] != 'arm':
                continue
            reference = StageReference(step, self._stage_cap(step['name']))
            governor = PhaseGovernor(reference, self.cfg['arrival_tolerance_rad'], **self.governor_options)
            governors[step['name']] = governor.metadata()
        sources = ('piper_oscbf/task_filter.py', 'piper_oscbf/task_geometry.py',
                   'piper_oscbf/task_reference.py', 'piper_oscbf/task_session.py',
                   'piper_oscbf/task_execution.py', 'piper_oscbf/device.py',
                   'scripts/run_pick_place.py')
        source_hashes = {name: _file_hash(name) for name in sources}
        mesh_hashes = {item['path']: item['sha256']
                       for part in geometry_metadata['mesh_coverage'] for item in part['meshes']}
        return {
            'path': {'steps': self.task['steps'], 'initial_q_rad': self.cfg['initial_q_rad'],
                     'transfer_via_tcp_m': self.cfg.get('transfer_via_tcp_m', [])},
            'geometry': {'scene': {k: self.cfg.get(k) for k in physical_keys},
                         'runtime_geometry': geometry_metadata,
                         'initial_payload_T_tcp': self.geometry.default_payload_T_tcp.tolist()},
            'parameters': {'kp_joint': f.kp_joint.tolist(), 'task_weights': f.task_weights.tolist(),
                'joint_velocity_limit_rad_s': self.cfg['joint_velocity_limit_rad_s'],
                'stage_velocity_limits_rad_s': self.cfg.get('stage_velocity_limits_rad_s', {}),
                'tracking_tolerance_rad': self.cfg['tracking_tolerance_rad'],
                'arrival_tolerance_rad': self.cfg['arrival_tolerance_rad'],
                'gripper_open_m': self.cfg['gripper_open_m'],
                'gripper_close_command_m': self.cfg['gripper_close_command_m'],
                'gripper_force_n': self.cfg['gripper_force_n'],
                'gripper_settle_tolerance_m': self.cfg.get('gripper_settle_tolerance_m',.0005),
                'velocity_quantization_rad_s': f.quantum, 'alpha_geometry': f.alpha_geometry,
                'alpha_joint': f.alpha_joint, 'objective_scale': f.objective_scale,
                'sigma_min_guard': f.sigma_min_guard, 'solver_tol': f.solver_tol,
                'residual_tol': f.residual_tol, 'physical_tolerance_m': f.physical_tolerance_m,
                'solver_max_iter': f.max_iter, 'reference_governor': governors},
            'controller': {'source_sha256': source_hashes,
                'nominal': f.metadata()['nominal'], 'quantization_policy': f.metadata()['quantization_policy'],
                'nominal_fast_path': f.metadata()['nominal_fast_path'],
                'method_difference': 'Only QP objective metric; method identity deliberately excluded from common hashes'},
            'constraints': {'labels': self.constraint_labels,
                'geometry_constraint_count': self.geometry_constraint_count,
                'alpha_geometry': f.alpha_geometry, 'alpha_joint': f.alpha_joint,
                'joint_lower_limits': list(self.geometry.tcp_robot.joint_lower_limits),
                'joint_upper_limits': list(self.geometry.tcp_robot.joint_upper_limits),
                'constraint_source_sha256': {name: source_hashes[name] for name in
                    ('piper_oscbf/task_filter.py', 'piper_oscbf/task_geometry.py')}},
            'model': {'urdf_sha256': {name: _file_hash(name) for name in
                ('models/piper_h_oscbf.urdf', 'models/piper_h_with_gripper.urdf',
                 'models/piper_h_fixed_gripper.urdf', 'models/piper_h_oscbf_model.json')},
                'collision_mesh_sha256': mesh_hashes,
                'tcp_ee_offset': np.asarray(self.geometry.tcp_robot.ee_offset).tolist()}}

    def _context(self, stage):
        flags = self.geometry.stage_defaults(stage)
        return {'carrying': self.carrying, 'support_contact': flags['support_contact'],
                'allow_finger_cube_contact': flags['allow_finger_cube_contact'],
                'payload_T_tcp': self.payload_T_tcp, 'world_cube_T': self.world_cube_T}

    def _failure(self, reason, diagnostics):
        self.failures.append({'reason': reason, 'diagnostics': jsonable(diagnostics)})
        raise TaskFilterError(reason, diagnostics)

    def filter_step(self, q, qref, qdref, width, cap, stage):
        if self._finished is not None:
            raise RuntimeError('Cannot add commands to a finished task session')
        if not np.isfinite(cap) or abs(float(cap)-self._stage_cap(stage)) > 1e-12:
            self._failure('stage_velocity_cap_mismatch', {'stage': stage, 'cap': cap,
                'recorded_cap': self._stage_cap(stage)})
        context = self._context(stage)
        try:
            result = self.filter.step(q, qref, qdref, width, cap, **context)
        except TaskFilterError as exc:
            diagnostics = {**exc.diagnostics, 'session_context': {
                'stage': stage, 'q': np.asarray(q), 'gripper_width_m': width, 'cap': cap, **context}}
            self._failure(exc.reason, diagnostics)
        result.update(q=np.asarray(q, float).copy(), q_ref=np.asarray(qref, float).copy(),
                      qd_ref=np.asarray(qdref, float).copy(), gripper_width_m=float(width),
                      velocity_cap_rad_s=float(cap), stage=str(stage), carrying=self.carrying,
                      support_contact=context['support_contact'])
        return result

    def validate_hold(self, q, width, cap, stage):
        result = self.filter_step(q, q, np.zeros(6), width, cap, stage)
        if np.any(np.asarray(result['u_sdk']) != 0.):
            self._failure('hold_requires_nonzero_command', result)
        return result

    def _gripper_transition_check(self, stage):
        if stage not in ('open', 'grasp', 'release'):
            self._failure('unknown_gripper_stage', {'stage': stage})
        if stage in ('open', 'grasp') and self.carrying:
            self._failure('unexpected_gripper_stage_while_carrying', {'stage': stage})
        if stage == 'release' and not self.carrying:
            self._failure('release_without_model_attachment', {'stage': stage})

    def check_gripper(self, q, start_width, end_width, stage, *, physical_tol=1e-8):
        self._gripper_transition_check(stage)
        if (not np.isfinite(physical_tol) or physical_tol < 0 or physical_tol > 1e-8
                or physical_tol > self.filter.physical_tolerance_m):
            raise ValueError('Gripper contact numerical tolerance must not exceed 1e-8 m or filter tolerance')
        flags = self.geometry.stage_defaults(stage)
        widths = np.linspace(float(start_width), float(end_width), 21)
        checks = [self.geometry.evaluate(q, stage, float(width), payload_T_tcp=self.payload_T_tcp,
            carrying=self.carrying, world_cube_T=self.world_cube_T,
            support_contact=flags['support_contact']) for width in widths]
        buffer = np.asarray(self.geometry.width_lipschitz_bounds()) * abs(end_width-start_width)/(2*(len(widths)-1))
        h = np.min([c['h'] for c in checks], axis=0)-buffer
        physical = np.min([c['physical_distance_m'] for c in checks], axis=0)-buffer
        active = checks[0]['active_mask']
        contact = np.asarray([name.startswith('table:payload/') for name in self.geometry.constraint_labels])
        contact &= bool(self.carrying and flags['support_contact'])
        allowed = np.where(contact, -physical_tol, 0.)
        bad = active & ((h < allowed) | (physical < allowed))
        report = {'stage': stage, 'q': np.asarray(q, float), 'start_width_m': float(start_width),
            'end_width_m': float(end_width), 'samples': len(widths),
            'physical_tol_m_numeric_contact_only': float(physical_tol),
            'minimum_h_lower_bound_m': float(h[active].min()),
            'minimum_physical_lower_bound_m': float(physical[active].min()),
            'h_lower_bound': h, 'physical_h_lower_bound': physical,
            'active_mask': active, 'numeric_contact_mask': contact,
            'violating_labels': [self.geometry.constraint_labels[i] for i in np.flatnonzero(bad)],
            'carrying': self.carrying, 'clear_for_fixed_q': not bool(np.any(bad)),
            'scope': 'Fixed arm q; known model geometry; every intermediate opening covered by row-wise Lipschitz bounds'}
        if np.any(bad):
            self._failure('gripper_sweep_not_clear', report)
        self.events.append({'event': 'gripper_sweep_checked', 'event_index': len(self.events),
            **{key: jsonable(report[key]) for key in ('stage', 'start_width_m', 'end_width_m',
                'minimum_h_lower_bound_m', 'minimum_physical_lower_bound_m', 'physical_tol_m_numeric_contact_only')}})
        return report

    def after_gripper(self, stage, q, width):
        self._gripper_transition_check(stage)
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.all(np.isfinite(q)) or not np.isfinite(width) or not 0 <= width <= .1:
            raise ValueError('Finite actual joint state and gripper opening required after gripper action')
        before = self.carrying
        if stage == 'grasp':
            self.payload_T_tcp = np.asarray(self.geometry.attach_cube(q, self.world_cube_T))
            self.carrying = True
        elif stage == 'release':
            self.world_cube_T = np.asarray(self.geometry.cube_world_transform(q, self.payload_T_tcp))
            self.carrying = False
        event = {'event': 'geometry_after_gripper', 'event_index': len(self.events), 'stage': stage,
            'carrying_before': before, 'carrying_after': self.carrying, 'q': q.tolist(),
            'measured_gripper_width_m': float(width), 'payload_T_tcp': self.payload_T_tcp.tolist(),
            'world_cube_T': self.world_cube_T.tolist(),
            'state_source': 'model FK plus scene/rigid-attachment assumption; not physical grasp or transfer confirmation'}
        self.events.append(event)
        return copy.deepcopy(event)

    def add_control(self, row, result):
        index = len(self.control)
        if row.get('control_row_index', index) != index or row.get('filter_row_index', index) != index:
            raise ValueError('Control/filter row indices are not aligned')
        row['control_row_index'] = row['filter_row_index'] = index
        arrays = {}
        for key in ARRAY_FIELDS:
            value = result.get(key)
            expected = (6,) if key in ('u_nom', 'u_qp', 'u_sdk') else (len(self.constraint_labels),)
            if value is None or np.asarray(value).shape != expected:
                raise ValueError(f'Missing or misaligned successful filter field: {key}')
            arrays[key] = np.asarray(value).copy()
        self.control.append({'row_index': index, 'stage': str(row.get('stage', result['stage'])),
            'monotonic_s': row.get('monotonic_s'), 'arrays': arrays})

    def add_hold(self, row, result):
        index = len(self.holds)
        row['hold_row_index'] = index
        raw = {key: result.get(key) for key in ARRAY_FIELDS}
        raw.update({key: result.get(key) for key in ('q', 'q_ref', 'qd_ref', 'gripper_width_m',
            'velocity_cap_rad_s', 'stage', 'carrying', 'support_contact', 'solver_status',
            'converged', 'iterations', 'qp_solved', 'sigma_min', 'min_h_m',
            'min_physical_distance_m', 'solve_ms', 'filter_active', 'max_applied_violation')})
        self.holds.append(jsonable({'hold_row_index': index, 'row': copy.deepcopy(row), **raw}))

    def add_schedule(self, stage, reference, states):
        if stage != reference.name:
            raise ValueError('Stage name differs from its reference')
        if stage in self.schedules:
            raise ValueError('A stage schedule was already recorded')
        states = list(states)
        if states:
            record = make_schedule_record(reference, states)
            record['phase_states'] = jsonable(states)
        else:
            record = {'schema': 1, 'stage': stage, 'reference_signature': reference.signature,
                      'record_status': 'not_started_no_phase_samples', 'elapsed_s': [], 'phase_s': []}
        self.schedules[stage] = record
        return {'stage': stage, 'record_status': record['record_status'], 'samples': len(states),
                'schedule_sha256': record.get('schedule_sha256')}

    def finish(self, out):
        if self._finished is not None:
            return copy.deepcopy(self._finished)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        n = len(self.control)
        arrays = {key: (np.stack([v['arrays'][key] for v in self.control]) if n else
                   np.empty((0, 6 if key in ('u_nom', 'u_qp', 'u_sdk') else len(self.constraint_labels)),
                            dtype=bool if key == 'constraint_mask' else float)) for key in ARRAY_FIELDS}
        arrays.update(control_row_index=np.arange(n, dtype=np.int64),
            stage=np.asarray([v['stage'] for v in self.control], dtype='U64'),
            monotonic_s=np.asarray([v['monotonic_s'] if v['monotonic_s'] is not None else np.nan
                                    for v in self.control]),
            constraint_labels=np.asarray(self.constraint_labels, dtype='U256'),
            geometry_constraint_count=np.asarray(self.geometry_constraint_count, dtype=np.int64))
        np.savez_compressed(out/'filter_constraints.npz', **arrays)
        with (out/'hold_safety.jsonl').open('w', encoding='utf-8') as stream:
            for record in self.holds:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        (out/'reference_schedules.json').write_text(json.dumps(jsonable({'schema': 1,
            'comparison_scope': 'Observed adaptive traces; same policy does not imply shared reference time',
            'schedules': list(self.schedules.values())}), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        alignment = {'filter_rows': n, 'control_csv_rows': None, 'status': 'CSV not yet written'}
        if (out/'control.csv').is_file():
            with (out/'control.csv').open() as stream:
                control_rows = list(csv.DictReader(stream))
            alignment['control_csv_rows'] = len(control_rows)
            indices_ok = all(int(r.get('filter_row_index', '-1')) == i
                             and int(r.get('control_row_index', '-1')) == i for i, r in enumerate(control_rows))
            alignment['status'] = 'matched' if len(control_rows) == n and indices_ok else 'MISMATCH'
        metadata = {**self.filter.metadata(), 'schema': 1, 'fingerprints': self.fingerprint,
            'fingerprint_payloads': self._fingerprint_payloads,
            'fingerprint_scope': 'Method-independent common task/model/controller/settings; actual attachment events and observed phase traces excluded',
            'geometry': self.geometry.metadata(), 'h_labels': self.h_labels,
            'constraint_labels': self.constraint_labels, 'geometry_constraint_count': self.geometry_constraint_count,
            'warmup': self.warmup, 'initial_hold_check': self.initial_hold,
            'control_records': n, 'hold_records': len(self.holds), 'control_alignment': alignment,
            'array_alignment': 'filter_constraints.npz row i corresponds only to control.csv row i; hold_safety.jsonl has independent indices',
            'nonfinite_json_encoding': 'NaN, Infinity and -Infinity are strings in failure diagnostics',
            'lifecycle_events': self.events, 'failures': self.failures,
            'final_model_state': {'carrying': self.carrying, 'payload_T_tcp': self.payload_T_tcp,
                                  'world_cube_T': self.world_cube_T},
            'object_transfer_confirmation': 'Not inferred by this module; requires operator or video evidence'}
        (out/'filter_metadata.json').write_text(json.dumps(jsonable(metadata), ensure_ascii=False,
                                                          indent=2, allow_nan=False)+'\n')
        self._finished = {'method': self.method, 'control_records': n, 'hold_records': len(self.holds),
            'schedule_stages': list(self.schedules), 'failure_count': len(self.failures),
            'fingerprints': self.fingerprint, 'control_alignment': alignment,
            'lifecycle_events': jsonable(self.events), 'final_model_carrying': self.carrying}
        return copy.deepcopy(self._finished)
