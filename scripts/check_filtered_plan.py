"""Essential static interpolation/geometry/QP checks, not a closed-loop simulation."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from piper_oscbf.preflight import plan_hash, source_hashes
from piper_oscbf.task_reference import StageReference
from piper_oscbf.task_session import TaskSession, jsonable

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--methods',nargs='+',choices=['cbf','oscbf'],default=['oscbf'])
    parser.add_argument('--speed-scale',type=float,default=1.)
    parser.add_argument('--descent-speed-scale',type=float,default=1.)
    args=parser.parse_args()
    task=json.loads(args.plan.read_text())
    proof={'status':'failed','scope':'Static saved-path/interpolant geometry and QP feasibility; no state integration, CAN or hardware',
           'plan_sha256':plan_hash(args.plan),'source_sha256':source_hashes(),
           'speed_scale':args.speed_scale,'descent_speed_scale':args.descent_speed_scale,
           'methods_checked':[],'method_results':{},'error':None}
    try:
        assert task['report']['status']=='planned'
        cfg=task['config'];saved=cfg['joint_velocity_limit_rad_s']
        cfg['joint_velocity_limit_rad_s']=saved*args.speed_scale
        cfg['stage_velocity_limits_rad_s']={s:saved*args.descent_speed_scale for s in ['approach','lower']}
        for method in args.methods:
            session=TaskSession(copy.deepcopy(task),method)
            q=np.asarray(cfg['initial_q_rad'],float);width=cfg['gripper_open_m']
            probe=task['report']['planning_tcp_probe']['pose']['pose']
            expected=np.eye(4)
            expected[:3,3]=[probe['position'][k] for k in ('x','y','z')]
            expected[:3,:3]=Rotation.from_quat([probe['orientation'][k] for k in ('x','y','z','w')]).as_matrix()
            actual=np.asarray(session.geometry.tcp_transform(q))
            fk_error=float(np.max(np.abs(actual-expected)))
            assert fk_error<1e-6, f'MoveIt/algorithm TCP mismatch {fk_error}'
            stages=[]
            for step in task['steps']:
                if step['kind']=='gripper':
                    session.check_gripper(q,width,step['width_m'],step['name'])
                    width=cfg['cube_size_m'] if step['name']=='grasp' else step['width_m']
                    session.after_gripper(step['name'],q,width)
                    continue
                if step['kind']!='arm':continue
                cap=cfg['stage_velocity_limits_rad_s'].get(step['name'],cfg['joint_velocity_limit_rad_s'])
                ref=StageReference(step,cap)
                # Include every saved knot and a <=10ms interpolation grid.
                phases=np.unique(np.r_[ref.times,np.arange(0,ref.duration,.01),ref.duration])
                metrics={'stage':step['name'],'samples':len(phases),'qp_solved_samples':0,
                         'min_sigma':float('inf'),'min_h_m':float('inf'),
                         'min_physical_distance_m':float('inf'),'min_applied_margin':float('inf'),
                         'max_step_ms':0.}
                for phase in phases:
                    q,qd=ref.sample(float(phase))
                    value=session.filter_step(q,q,qd,width,cap,step['name'])
                    metrics['qp_solved_samples']+=int(value['qp_solved'])
                    for target,key in [('min_sigma','sigma_min'),('min_h_m','min_h_m'),
                                       ('min_physical_distance_m','min_physical_distance_m')]:
                        metrics[target]=min(metrics[target],float(value[key]))
                    metrics['min_applied_margin']=min(metrics['min_applied_margin'],float(np.min(value['applied_margin'])))
                    metrics['max_step_ms']=max(metrics['max_step_ms'],value['solve_ms'])
                session.validate_hold(q,width,cap,step['name'])
                stages.append(metrics)
            proof['methods_checked'].append(method)
            proof['method_results'][method]={'stages':stages,'tcp_fk_max_abs_difference':fk_error,
                'warmup':session.warmup,'initial_hold':session.initial_hold,
                'gripper_width_scope':'planned cube-width nominal during arm motion; full commanded opening sweep checked; hardware reads actual width'}
        proof['status']='passed'
    except Exception as exc:
        proof['error']=f'{type(exc).__name__}: {exc}'
        if hasattr(exc,'diagnostics'):proof['failure_diagnostics']=jsonable(exc.diagnostics)
    (args.plan.parent/'filter_preflight.json').write_text(json.dumps(jsonable(proof),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ['status','methods_checked','error','method_results']},ensure_ascii=False,indent=2))
    return 0 if proof['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
