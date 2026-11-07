"""Bounded, text-only checks of the prepared full-task safety filter; no CAN."""
import copy
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from piper_oscbf.task_filter import TaskSafetyFilter, TaskFilterError, quantize_sdk


def subset(result):
    names=('method','sigma_min','metric_condition','converged','iterations','qp_solved','solver_status',
           'filter_active','solve_ms','min_h_m','min_physical_distance_m','max_applied_violation',
           'nominal_constraint_violation','u_nom','u_qp','u_sdk','position_error_m','orientation_error_rad')
    return {key:result[key].tolist() if isinstance(result[key],np.ndarray) else result[key] for key in names}


def main():
    out=ROOT/'results/task_filter_check_20261107'
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/pick_place_obstacle_candidate.json').read_text())
    cfg['joint_velocity_limit_rad_s']=.10
    cfg['safety_filter']={'alpha_geometry':2.,'alpha_joint':3.,'kp_joint':2.,'task_weights':[1.]*6,
        'objective_scale':100.,'sigma_min_guard':.10,'velocity_quantization_rad_s':.001,
        'solver_tol':1e-6,'residual_tol':1e-6,'physical_tolerance_m':1e-8,'solver_max_iter':50}
    design=json.loads((ROOT/'results/task_obstacle_design_20261107/scene_proposal.json').read_text())
    q_via=np.asarray(design['via_ik_and_clearances']['q_rad'])
    q_start=np.asarray(cfg['initial_q_rad'])
    task=json.loads((ROOT/'results/pick_place_plan_20261104_224921_126900/task.json').read_text())
    lower=next(step for step in task['steps'] if step.get('name')=='lower')['trajectory']['joint_trajectory']
    q_contact=np.asarray([lower['points'][-1]['positions'][lower['joint_names'].index(f'joint{j}')] for j in range(1,7)])
    report={'hardware_connected':False,'images_generated':False,'checks':{},'timings':{},
            'settings':cfg['safety_filter'],'candidate_config_used':cfg,
            'versions':{name:version(name) for name in ('jax','jaxlib','numpy','qpax')},
            'source_sha256':{str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                for path in (Path('piper_oscbf/task_filter.py'),Path('piper_oscbf/task_geometry.py'),
                             Path('scripts/check_task_filter.py'))}}
    filters={}
    for scene in ('no_obstacle','candidate_obstacle'):
        scene_cfg=copy.deepcopy(cfg)
        if scene=='no_obstacle':scene_cfg['obstacles']=[]
        for method in ('cbf','oscbf'):
            print('CHECKING',scene,method,flush=True)
            safety=TaskSafetyFilter(scene_cfg,method)
            key=f'{scene}/{method}'
            report['checks'][key]={'warmup':safety.warmup(q_start,.10),'metadata':safety.metadata()}
            filters[key]=safety
            normal=safety.step(q_start,q_start,np.zeros(6),.04,.10)
            assert not normal['qp_solved'] and normal['solver_status']=='nominal_quantized_feasible'
            assert np.array_equal(normal['u_sdk'],np.zeros(6))
            assert normal['h'].shape==normal['b_physical'].shape==(len(safety.constraint_labels),)
            assert normal['G'].shape==(len(safety.constraint_labels),6)
            report['checks'][key]['prepared_hold']=subset(normal)
            # Model table-contact negative residual is retained, not clamped.
            contact=safety.step(q_contact,q_contact,np.zeros(6),.02,.10,carrying=True,support_contact=True,
                payload_T_tcp=safety.geometry.default_payload_T_tcp,
                world_cube_T=safety.geometry.world_cube_transform(True),allow_finger_cube_contact=True)
            assert not contact['qp_solved'] and np.array_equal(contact['u_sdk'],np.zeros(6))
            assert contact['min_h_m'] >= -1e-8
            report['checks'][key]['exact_zero_support_contact']=subset(contact)
            try:
                safety.step(np.zeros(6),np.zeros(6),np.zeros(6),.04,.10)
            except TaskFilterError as error:
                assert error.reason=='outside_audited_full_rank_region',error.reason
                report['checks'][key]['zero_pose_rejected']={'reason':error.reason,'sigma_min':error.diagnostics['sigma_min']}
            else:raise AssertionError('Singular zero pose was accepted')
            case=(q_via if scene=='candidate_obstacle' else q_start)
            values=[]
            for i in range(30):
                value=safety.step(case,case,np.zeros(6),.02,.10,carrying=True,support_contact=False,
                    payload_T_tcp=safety.geometry.default_payload_T_tcp,
                    world_cube_T=safety.geometry.world_cube_transform(),allow_finger_cube_contact=False)
                assert np.array_equal(value['u_sdk'],quantize_sdk(value['u_qp']))
                assert np.max(np.abs(value['u_sdk']))<=.10+1e-12
                assert value['applied_margin'].min()>=-safety.residual_tol
                values.append(value['solve_ms'])
            if scene=='candidate_obstacle':
                assert value['qp_solved'] and value['filter_active']
                assert value['min_h_m']<0 and value['min_physical_distance_m']>0
            report['checks'][key]['numeric_case']=subset(value)
            report['timings'][key]={'calls':len(values),'ms_median_p95_max':np.percentile(values,[50,95,100]).tolist(),
                                   'scope':'whole step incl. model/Jacobian/sync/solve/check; not full control loop or USB timing'}
            print(key,'timing',report['timings'][key],flush=True)
            (out/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # Both methods must expose exactly the same constraints and nominal input.
    q_ref=q_via+np.array([.002,-.003,.001,0.,-.002,.001])
    qd_ref=np.array([.015,-.020,.010,.005,-.010,.003])
    a=filters['candidate_obstacle/cbf'].step(q_via,q_ref,qd_ref,.02,.10,carrying=True)
    b=filters['candidate_obstacle/oscbf'].step(q_via,q_ref,qd_ref,.02,.10,carrying=True)
    for key in ('u_nom','h','physical_h','G','b_physical','b_robust'):
        assert np.array_equal(a[key],b[key]),key
    assert not np.allclose(a['metric'],b['metric'])
    expected=qd_ref+2*(q_ref-q_via)
    expected/=max(1.,np.max(np.abs(expected))/.10)
    assert np.allclose(a['u_nom'],expected,atol=1e-15,rtol=0)
    report['checks']['shared_inputs_and_constraints']={'passed':True,'maximum_nominal_difference':0.,
        'nominal_rad_s':a['u_nom'].tolist(),
        'maximum_applied_difference_rad_s':float(np.max(np.abs(a['u_sdk']-b['u_sdk'])))}
    report['status']='passed'
    report['scope']='Isolated filter checks, not whole-task trajectory feasibility or hardware success.'
    (out/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('TASK_FILTER_CHECKS_PASSED',flush=True)


if __name__=='__main__':
    main()
