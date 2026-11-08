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
                'velocity_quantization_rad_s': f.quantum, 'alpha_geometry': f.alpha_geometry,
                'alpha_joint': f.alpha_joint, 'objective_scale': f.objective_scale,
                'solver_tol':f.solver_tol,'residual_tol':f.residual_tol,
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

    def observe_hold(self, q, width, cap, stage):
        context=self._context(stage)
        result=self.filter.observe_zero(q,width,cap,**context)
        result.update(q=np.asarray(q,float).copy(),q_ref=np.asarray(q,float).copy(),
            qd_ref=np.zeros(6),gripper_width_m=float(width),velocity_cap_rad_s=float(cap),
            stage=str(stage),carrying=self.carrying,support_contact=context['support_contact'])
        return result


    def after_gripper(self, stage, q, width):
        q = np.asarray(q, float)
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
        index=len(self.control)
        row['control_row_index']=row['filter_row_index']=index
        arrays={key:np.asarray(result[key]).copy() for key in ARRAY_FIELDS}
        self.control.append({'row_index':index,'stage':str(row.get('stage',result['stage'])),
            'monotonic_s':row.get('monotonic_s'),'arrays':arrays})

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
            'warmup': self.warmup,
            'hold_semantics':'Recorded zero-velocity requests, not a fixed-position guarantee; no gripper gate',
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
