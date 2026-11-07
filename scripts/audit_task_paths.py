"""Text-only audit of saved Piper paths, without SDK/CAN/ROS/GUI access.

Samples every saved point and each actual linear interpolation segment densely;
refines determinant sign changes and sampled low-sigma regions. This finite
audit is evidence about the saved reference, not a proof for arbitrary tracking
errors or a geometric collision certification.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.transform import Rotation
from oscbf.core.manipulator import Manipulator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANS = [
    'results/pick_place_plan_20261104_224921_126900/task.json',
    'results/pick_place_plan_20261104_232956_473856/task.json',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_arrays(step, cap):
    """Mirror run_pick_place.path_arrays without importing a hardware entry."""
    traj = step['trajectory']['joint_trajectory']
    order = [traj['joint_names'].index(f'joint{i}') for i in range(1, 7)]
    q = np.asarray([[v['positions'][i] for i in order] for v in traj['points']])
    t = np.asarray([v['time_from_start']['sec'] + v['time_from_start']['nanosec'] * 1e-9
                    for v in traj['points']])
    dt = np.maximum(np.diff(t), np.abs(np.diff(q, axis=0)).max(axis=1) / (cap * .5))
    return np.r_[0., np.cumsum(np.maximum(dt, .01))], q


def json_clean(v):
    if isinstance(v, dict): return {k: json_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [json_clean(x) for x in v]
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.ndarray): return json_clean(v.tolist())
    return v


def working_pose_study(out, args):
    """Study a candidate preparation region; does not construct a motion plan."""
    source = ROOT / DEFAULT_PLANS[0]
    task = json.loads(source.read_text())
    offset = np.eye(4); offset[:3, 3] = task['config']['tcp_offset_m']
    metadata = json.loads((ROOT / 'models/piper_h_oscbf_model.json').read_text())
    robot = Manipulator.from_urdf(str(ROOT / 'models/piper_h_oscbf.urdf'),
                                 ee_offset=np.asarray(metadata['ee_offset']) @ offset)
    batch = jax.jit(jax.vmap(robot.ee_jacobian))
    def endpoint(name):
        step = next(v for v in task['steps'] if v.get('name') == name)
        return path_arrays(step, .3)[1][-1]
    prep, left = endpoint('pregrasp'), endpoint('retreat')
    connector = np.linspace(left, prep, 2001)
    ss = np.linalg.svd(np.asarray(batch(jnp.asarray(connector))), compute_uv=False)
    work = []
    for step in task['steps']:
        if step['kind'] != 'arm' or step['name'] in ('pregrasp', 'return_initial'): continue
        cap = task['config']['joint_velocity_limit_rad_s'] * (
            args.descent_speed_scale if step['name'] in ('approach', 'lower') else args.speed_scale)
        times, points = path_arrays(step, cap)
        for i, (qa, qb) in enumerate(zip(points, points[1:])):
            divisions = max(1, math.ceil((times[i+1]-times[i]) / args.sample_period),
                            math.ceil(np.max(np.abs(qb-qa)) / args.max_joint_step))
            work.extend(np.linspace(qa, qb, divisions+1))
    qs = np.vstack([work, connector])
    corners = np.asarray(list(itertools.product([-.01, .01], repeat=6)))
    perturbed = (qs[:, None, :] + corners[None, :, :]).reshape(-1, 6)
    minimum, max_condition, signs, minimum_q = math.inf, 0., set(), None
    for start in range(0, len(perturbed), 4096):
        chunk = perturbed[start:start+4096]
        matrices = np.asarray(batch(jnp.asarray(chunk)))
        sigma = np.linalg.svd(matrices, compute_uv=False)
        at = int(np.argmin(sigma[:, -1]))
        if sigma[at, -1] < minimum:
            minimum, minimum_q = float(sigma[at, -1]), chunk[at].tolist()
        max_condition = max(max_condition, float(((sigma[:, 0]/sigma[:, -1])**2).max()))
        signs.update(np.sign(np.linalg.det(matrices)).astype(int).tolist())
    report = {'source_plan': DEFAULT_PLANS[0], 'source_sha256': sha(source),
        'proposed_initial_and_return_q_rad': prep.tolist(), 'proposed_initial_and_return_q_deg': np.rad2deg(prep).tolist(),
        'proposed_tcp_xyz_m': np.asarray(robot.ee_transform(jnp.asarray(prep)))[:3, 3].tolist(),
        'connector_scope': 'Dense straight joint interpolation only; not a saved MoveIt plan and not collision-checked. Replan with MoveIt, then audit actual generated plan before execution.',
        'left_retreat_to_right_pregrasp_connector_samples': len(connector),
        'connector_min_sigma': float(ss[:, -1].min()),
        'connector_metric_condition_max': float(((ss[:, 0]/ss[:, -1])**2).max()),
        'combined_working_reference_samples': len(qs), 'joint_perturbation_box_halfwidth_rad': .01,
        'perturbation_corner_samples': len(perturbed), 'corner_sample_min_sigma': minimum,
        'corner_sample_max_metric_condition': max_condition, 'corner_sample_det_signs': sorted(signs),
        'minimum_corner_q_rad': minimum_q, 'proposed_runtime_sigma_floor': .1,
        'sigma_floor_status': 'Engineering guard chosen below observed reference/corner samples; not a manufacturer limit or analytic proof over all tracking errors. Recheck on new MoveIt path and obstacle-filtered trajectory.',
        'remaining_requirement': 'Separately operator-start baseline preparation from actual pose. Filtered task starts/returns at prepared posture; preparation rank crossing is not full-rank OSCBF.'}
    (out/'working_pose_proposal.json').write_text(json.dumps(report, indent=2)+'\n')
    candidates = [[0,60,-70,0,60,0], [0,60,-90,0,60,0], [0,70,-80,0,80,0],
                  [0,80,-80,0,85,0], [0,94.11,-72.94,0,73.83,0],
                  [0,50,-70,0,65,0], [0,30,-90,0,45,0], [0,20,-50,0,30,0]]
    alternate = []
    for degrees in candidates:
        q = np.deg2rad(degrees)
        J = np.asarray(robot.ee_jacobian(jnp.asarray(q)))
        item = {'q_deg': degrees, 'sigma_min': float(np.linalg.svd(J, compute_uv=False)[-1]),
                'det_J': float(np.linalg.det(J)), 'tcp_xyz_m': np.asarray(robot.ee_transform(jnp.asarray(q)))[:3, 3].tolist(),
                'straight_joint_connector_checks': []}
        for goal in (prep, left):
            matrices = np.asarray(batch(jnp.asarray(np.linspace(q, goal, 2001))))
            sigma = np.linalg.svd(matrices, compute_uv=False)
            determinant = np.linalg.det(matrices)
            item['straight_joint_connector_checks'].append({'sigma_min': float(sigma[:, -1].min()),
                                                            'det_sign_flip': bool(determinant[0]*determinant[-1] < 0)})
        alternate.append(item)
    (out/'alternate_pose_candidates.json').write_text(json.dumps(alternate, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', nargs='+', default=DEFAULT_PLANS)
    parser.add_argument('--out', default='results/task_path_audit_20261107')
    parser.add_argument('--speed-scale', type=float, default=1.5)
    parser.add_argument('--descent-speed-scale', type=float, default=1.5)
    parser.add_argument('--sample-period', type=float, default=.01)
    parser.add_argument('--max-joint-step', type=float, default=.001)
    parser.add_argument('--sigma-guard', type=float, default=.005)
    parser.add_argument('--working-pose-study', action='store_true', help='Also audit the proposed right-pregrasp working region; never creates a hardware path')
    args = parser.parse_args()
    if min(args.speed_scale, args.descent_speed_scale, args.sample_period,
           args.max_joint_step, args.sigma_guard) <= 0: parser.error('All audit parameters must be positive.')
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / 'models/piper_h_oscbf.urdf'
    metadata_path = ROOT / 'models/piper_h_oscbf_model.json'
    metadata = json.loads(metadata_path.read_text())
    executor_path = ROOT / 'scripts/run_pick_place.py'
    executor_tree = ast.parse(executor_path.read_text())
    reference_function = next(v for v in executor_tree.body if isinstance(v, ast.FunctionDef) and v.name == 'path_arrays')
    reference_namespace = {'np': np, 'JOINTS': [f'joint{i}' for i in range(1, 7)]}
    exec(compile(ast.Module(body=[reference_function], type_ignores=[]), str(executor_path), 'exec'), reference_namespace)
    flange_offset = np.asarray(metadata['ee_offset'])
    flange = Manipulator.from_urdf(str(model_path), ee_offset=flange_offset)
    plans = []
    candidate_reports = None
    for name in args.plans:
        source = Path(name)
        if not source.is_absolute(): source = ROOT / source
        task = json.loads(source.read_text())
        cfg = task['config']
        tcp_shift = np.eye(4)
        tcp_shift[:3, 3] = cfg['tcp_offset_m']
        robot = Manipulator.from_urdf(str(model_path), ee_offset=flange_offset @ tcp_shift)
        jac = jax.jit(robot.ee_jacobian)
        fk = jax.jit(robot.ee_transform)
        batch_jac = jax.jit(jax.vmap(robot.ee_jacobian))
        batch_fk = jax.jit(jax.vmap(robot.ee_transform))
        lower, upper = np.asarray(robot.joint_lower_limits), np.asarray(robot.joint_upper_limits)
        alpha, half_quantum = 3., .0005
        candidates = {'zero': [0, 0, 0, 0, 0, 0],
                      'historical_preparation': [0, 20, -50, 0, 30, 0],
                      'plane_experiment_start': [0, 30, -90, 0, 45, 0]}
        candidate_reports = {}
        for key, degrees in candidates.items():
            q = np.deg2rad(degrees)
            J = np.asarray(jac(jnp.asarray(q)))
            s = np.linalg.svd(J, compute_uv=False)
            candidate_reports[key] = {'q_deg': degrees, 'sigma': s,
                'metric_condition_number': (s[0] / s[-1])**2,
                'rank_at_1e-10': int((s > 1e-10).sum()),
                'joint_margin_rad': float(np.minimum(q - lower, upper - q).min()),
                'tcp_position_m': np.asarray(fk(jnp.asarray(q)))[:3, 3]}
        fd_errors = []
        for degrees in candidates.values():
            q = np.deg2rad(degrees)
            J = np.asarray(jac(jnp.asarray(q)))
            numerical = np.zeros((6, 6))
            for j in range(6):
                dq = np.eye(6)[j] * 1e-6
                plus, minus = [np.asarray(fk(jnp.asarray(x))) for x in (q + dq, q - dq)]
                numerical[:3, j] = (plus[:3, 3] - minus[:3, 3]) / 2e-6
                numerical[3:, j] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / 2e-6
            fd_errors.append(float(np.abs(J - numerical).max()))
        assert max(fd_errors) < 1e-6, fd_errors
        stage_reports, all_rows, refinements = [], [], []
        for step in task['steps']:
            if step['kind'] != 'arm': continue
            cap = cfg['joint_velocity_limit_rad_s'] * (
                args.descent_speed_scale if step['name'] in ('approach', 'lower') else args.speed_scale)
            times, points = path_arrays(step, cap)
            actual_times, actual_points = reference_namespace['path_arrays'](step, cap)
            assert np.array_equal(times, actual_times) and np.array_equal(points, actual_points)
            qs, qds, sample_meta = [], [], []
            if len(points) == 1:
                # A zero-length pregrasp is still a real required hold pose.
                qs.append(points[0]); qds.append(np.zeros(6))
                sample_meta.append((0, 0., 0., True))
            for i, (qa, qb) in enumerate(zip(points, points[1:])):
                duration = times[i + 1] - times[i]
                divisions = max(1, math.ceil(duration / args.sample_period),
                                math.ceil(np.max(np.abs(qb - qa)) / args.max_joint_step))
                for frac in np.linspace(0., 1., divisions + 1):
                    qs.append(qa * (1 - frac) + qb * frac)
                    if frac == 1.:
                        # Executor uses searchsorted(side='right') at an internal
                        # point, and zero feed-forward at the final reference time.
                        qds.append((points[i+2] - qb) / (times[i+2] - times[i+1])
                                   if i + 2 < len(points) else np.zeros(6))
                    else:
                        qds.append((qb - qa) / duration)
                    sample_meta.append((i, frac, times[i] + frac * duration, frac in (0., 1.)))
            qs, qds = np.asarray(qs), np.asarray(qds)
            Js = np.asarray(batch_jac(jnp.asarray(qs)))
            Ts = np.asarray(batch_fk(jnp.asarray(qs)))
            sigmas = np.linalg.svd(Js, compute_uv=False)
            determinants = np.linalg.det(Js)
            margins = np.minimum(qs - lower, upper - qs).min(axis=1)
            bounds_low = np.maximum(-cap + half_quantum, -alpha * (qs - lower) + half_quantum)
            bounds_high = np.minimum(cap - half_quantum, alpha * (upper - qs) - half_quantum)
            feasibility = (bounds_high - bounds_low).min(axis=1)
            nominal_margins = np.minimum(qds + alpha * (qs - lower),
                                        -qds + alpha * (upper - qs)).min(axis=1)
            quantized = np.round(qds, 3)
            quantized_margins = np.minimum(quantized + alpha * (qs - lower),
                                          -quantized + alpha * (upper - qs)).min(axis=1)
            minimum = int(np.argmin(sigmas[:, -1]))
            stage_summary = {'stage': step['name'], 'saved_points': len(points), 'samples': len(qs),
                'full_rate_reference_duration_s': float(times[-1]),
                'sigma_min': float(sigmas[minimum, -1]),
                'minimum_at': {'segment': sample_meta[minimum][0], 'fraction': sample_meta[minimum][1],
                               'reference_time_s': sample_meta[minimum][2], 'q_deg': np.rad2deg(qs[minimum])},
                'rank_deficient_samples_at_1e-10': int((sigmas[:, -1] < 1e-10).sum()),
                'samples_below_sigma_guard': int((sigmas[:, -1] < args.sigma_guard).sum()),
                'metric_condition_max': float(np.max((sigmas[:, 0] / sigmas[:, -1])**2)),
                'min_joint_margin_rad': float(margins.min()),
                'minimum_joint_speed_feasible_width_rad_s': float(feasibility.min()),
                'min_reference_joint_cbf_margin_rad_s': float(nominal_margins.min()),
                'min_quantized_reference_joint_cbf_margin_rad_s': float(quantized_margins.min()),
                'reference_joint_cbf_violating_samples': int((nominal_margins < -1e-9).sum()),
                'max_reference_speed_rad_s': float(np.abs(qds).max())}
            stage_reports.append(stage_summary)
            for k, (i, frac, phase, saved) in enumerate(sample_meta):
                all_rows.append({'stage': step['name'], 'segment': i, 'fraction': frac,
                    'reference_time_s': phase, 'saved_point': int(saved),
                    'sigma_min': sigmas[k, -1], 'sigma_max': sigmas[k, 0],
                    'metric_condition': (sigmas[k, 0] / sigmas[k, -1])**2,
                    'joint_margin_rad': margins[k], 'joint_speed_feasible_width_rad_s': feasibility[k],
                    'reference_joint_cbf_margin_rad_s': nominal_margins[k],
                    'quantized_reference_joint_cbf_margin_rad_s': quantized_margins[k],
                    **{f'q{j+1}_rad': qs[k, j] for j in range(6)},
                    'tcp_x_m': Ts[k, 0, 3], 'tcp_y_m': Ts[k, 1, 3], 'tcp_z_m': Ts[k, 2, 3]})
            # Refine every determinant sign crossing of consecutive interpolation
            # samples, and each segment containing a below-guard sampled minimum.
            seen = set()
            for k in range(len(qs) - 1):
                i, fa = sample_meta[k][:2]
                i2, fb = sample_meta[k+1][:2]
                if i != i2 or fb <= fa: continue
                if determinants[k] * determinants[k+1] < 0:
                    qa, qb = points[i:i+2]
                    fun = lambda f: float(np.linalg.det(np.asarray(jac(jnp.asarray(qa*(1-f)+qb*f)))))
                    f = brentq(fun, fa, fb, xtol=1e-13)
                    q = qa*(1-f)+qb*f
                    sv = np.linalg.svd(np.asarray(jac(jnp.asarray(q))), compute_uv=False)
                    key = (step['name'], i, round(f, 8))
                    if key not in seen:
                        seen.add(key); refinements.append({'stage': step['name'], 'segment': i, 'fraction': f,
                            'reason': 'determinant_sign_crossing', 'sigma_min': sv[-1], 'q_deg': np.rad2deg(q)})
            for i in sorted({sample_meta[k][0] for k in np.flatnonzero(sigmas[:, -1] < args.sigma_guard)}):
                qa, qb = points[i:i+2]
                fun = lambda f: float(np.linalg.svd(np.asarray(jac(jnp.asarray(qa*(1-f)+qb*f))), compute_uv=False)[-1])
                opt = minimize_scalar(fun, bounds=(0., 1.), method='bounded', options={'xatol': 1e-12})
                best = min([(0., fun(0.)), (1., fun(1.)), (float(opt.x), float(opt.fun))], key=lambda x: x[1])
                refinements.append({'stage': step['name'], 'segment': i, 'fraction': best[0],
                    'reason': 'low_sigma_segment_minimization', 'sigma_min': best[1],
                    'q_deg': np.rad2deg(qa*(1-best[0])+qb*best[0])})
        stem = source.parent.name
        with (out / f'{stem}_samples.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
        plan_result = {'plan': str(source.relative_to(ROOT)), 'sha256': sha(source),
            'tcp_offset_m': cfg['tcp_offset_m'], 'tcp_calibrated': False,
            'speed_scale': args.speed_scale, 'descent_speed_scale': args.descent_speed_scale,
            'saved_initial_q_deg': np.rad2deg(cfg['initial_q_rad']),
            'finite_difference_jacobian_max_abs_errors': fd_errors,
            'stages': stage_reports, 'refined_low_rank_points': refinements,
            'all_joint_speed_boxes_feasible': all(s['minimum_joint_speed_feasible_width_rad_s'] >= 0 for s in stage_reports),
            'global_sampled_sigma_min': min(s['sigma_min'] for s in stage_reports),
            'global_refined_sigma_min': min((r['sigma_min'] for r in refinements), default=min(s['sigma_min'] for s in stage_reports)),
            'global_min_joint_margin_rad': min(s['min_joint_margin_rad'] for s in stage_reports)}
        plans.append(plan_result)
        print(stem, json.dumps(json_clean({k:plan_result[k] for k in ['global_sampled_sigma_min','global_refined_sigma_min','global_min_joint_margin_rad','all_joint_speed_boxes_feasible']})), flush=True)
        for s in stage_reports: print(s['stage'], 'sigma_min=',s['sigma_min'], 'joint_margin=',s['min_joint_margin_rad'], 'joint_cbf_violations=',s['reference_joint_cbf_violating_samples'], flush=True)
    report = {'created_at': datetime.now().astimezone().isoformat(), 'scope': 'offline_reference_path_audit_no_hardware',
        'urdf': str(model_path.relative_to(ROOT)), 'urdf_sha256': sha(model_path),
        'executor_sha256': sha(executor_path), 'path_arrays_exact_match_verified': True,
        'parameters': vars(args), 'task_metric': 'J_tcp.T @ J_tcp; Wo=I, N=0 full-rank square specialization',
        'conditioning_units': 'linear rows m/rad, angular rows rad/rad; conditioning depends on weights/units',
        'joint_cbf_alpha_per_s': 3., 'quantum_rad_s': .001, 'joint_speed_feasibility_scope': 'Joint-limit CBF and velocity box only, no obstacle/distance constraints; nominal margins assume actual state equals reference (tracking correction zero)',
        'sampling_scope': 'Every saved point, every segment at <=10 ms reference spacing and <=0.001 rad maximum joint increment; adaptive reference phases remain continuous within these same segments. Finite sampling is not a global proof.',
        'candidate_postures': candidate_reports, 'plans': plans,
        'paper_reference': 'Morton and Pavone, OSCBF arXiv v3, IV-B, equations (18)-(28), PDF page 4',
        'paper_rank_note': 'Eq(25) includes N.T Wj.T Wj N. N=0 only holds for full-rank square J. At rank loss, simply retaining J.T J is singular. A projector metric at exact rank loss does not by itself validate near-singular nominal inverse or rank changes.'}
    (out / 'audit.json').write_text(json.dumps(json_clean(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if args.working_pose_study: working_pose_study(out, args)
    print('AUDIT', out/'audit.json')


if __name__ == '__main__': main()
