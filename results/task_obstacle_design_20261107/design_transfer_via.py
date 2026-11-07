"""IK-only interior-reference candidates; not MoveIt or hardware validation."""
import copy
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import jax
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from piper_oscbf.task_geometry import TaskGeometry

OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--obstacle-x',type=float,default=.21)
parser.add_argument('--gap',type=float,default=.01)
parser.add_argument('--output-stem',default='transfer_via_candidates')
args=parser.parse_args()
cfg=json.loads((ROOT/'config/pick_place.json').read_text())
cfg['obstacles']=[{'name':'cylinder_box_candidate','center_m':[args.obstacle_x,0.,.028],'size_m':[.07,.07,.08]}]
cfg['runtime_geometry']={'obstacle_clearance_m':args.gap}
g=TaskGeometry(cfg)
task=json.loads((ROOT/'results/pick_place_plan_20261104_224921_126900/task.json').read_text())
arms={s['name']:s for s in task['steps'] if s['kind']=='arm'}
point=lambda name:np.asarray(arms[name]['trajectory']['joint_trajectory']['points'][-1]['positions'])
payload=np.asarray(g.attach_cube(point('approach')))
points=np.asarray([p['positions'] for p in arms['transfer']['trajectory']['joint_trajectory']['points']])
seed=points[len(points)//2]
R=Rotation.from_euler('xyz',[0.,np.pi,0.]).as_matrix()
robot=g.tcp_robot
lo=np.asarray(robot.joint_lower_limits)+1e-5;hi=np.asarray(robot.joint_upper_limits)-1e-5
fk=jax.jit(g.tcp_transform)
J=jax.jit(g.tcp_jacobian)
barriers=jax.jit(g.barrier_values)
fk(seed).block_until_ready()
finger=np.array([name.split('/')[0] in ('gripper_link1','gripper_link2') for name in g.sphere_labels])
rows=[]
for x in [.245,.25,.255,.26,.265,.27]:
    for z in [.095,.100,.105,.110,.115]:
        desired=np.array([x,0.,z])
        def residual(q):
            t=np.asarray(fk(q));rv=Rotation.from_matrix(R.T@t[:3,:3]).as_rotvec()
            return np.r_[t[:3,3]-desired,.15*rv]
        fit=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=150)
        q=fit.x;tf=np.asarray(fk(q))
        physical=np.asarray(barriers(q,.0275,True,False,payload,None,False,0.)).copy()
        # Conservative interval for any gripper opening in [15,40] mm.
        physical[:len(finger)][finger]-=.00625
        h=physical.copy();h[:len(finger)+1]-=args.gap
        actual_opening_clearances=[float(np.asarray(barriers(q,float(w),True,False,payload,None,False,0.))[:len(finger)+1].min())
                                   for w in np.linspace(.015,.04,6)]
        width_interval_upper_clearance=max(actual_opening_clearances)+(.04-.015)/(4*5)
        k=int(np.argmin(physical));sv=np.linalg.svd(np.asarray(J(q)),compute_uv=False)
        center=np.asarray(g.cube_world_transform(q,payload))[:3,3]
        payload_idx=len(finger)
        error_position=float(np.linalg.norm(tf[:3,3]-desired))
        error_angle=float(Rotation.from_matrix(R.T@tf[:3,:3]).magnitude())
        row={'tcp_x_m':x,'tcp_y_m':0.,'tcp_z_m':z,'q_rad':q.tolist(),
             'q_deg':np.rad2deg(q).tolist(),'ik_position_error_m':error_position,'ik_angle_error_rad':error_angle,
             'sigma_min':float(sv[-1]),'joint_margin_rad':float(np.min(np.r_[q-lo,hi-q])),
             'min_physical_clearance_m':float(physical[k]),'worst_constraint':g.constraint_labels[k],
             'minimum_virtual_h_m':float(h.min()),'payload_center_m':center.tolist(),
             'payload_physical_clearance_m':float(physical[payload_idx]),
             'payload_clearance_with_5mm_center_uncertainty_m':float(physical[payload_idx]-.005),
             'actual_opening_obstacle_clearance_range_m':[min(actual_opening_clearances),max(actual_opening_clearances)],
             'width_interval_lipschitz_upper_clearance_m':width_interval_upper_clearance,
             'virtual_activation_for_entire_opening_interval':width_interval_upper_clearance<args.gap-.001,
             'virtual_activation_for_all_checked_openings':max(actual_opening_clearances)<args.gap-.001,
             'qualified_ik_candidate':bool(error_position<1e-5 and error_angle<1e-4 and sv[-1]>.02
                and .005<=physical.min()<=args.gap-.001 and h.min()<-.001)}
        rows.append(row)
eligible=[r for r in rows if r['qualified_ik_candidate']]
report={'status':'ik_candidates_only','hardware_executed':False,'MoveIt_path_validated':False,
        'box_center_candidate_m':[args.obstacle_x,0.,.028],'box_size_envelope_m':[.07,.07,.08],
        'virtual_clearance_m':args.gap,'actual_xy_status':'not_measured_or_field_confirmed',
        'reference_note':'Via must be an interior reference point in a single transfer stage; it intentionally lies inside the virtual buffer but outside physical geometry, not an achievable required stop.',
        'held_finger_width_envelope_m':[.015,.04],
        'candidates':rows,'qualified_candidates':eligible,
        'next_checks':['MoveIt plan through a candidate using the physical box envelope',
            'Check entire interpolated path geometry and rank',
            'Run both filters with identical nominal reference and explicit phase schedule',
            'Confirm actual box/cylinder center, dimensions and upright placement before hardware']}
(OUT/(args.output_stem+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (OUT/(args.output_stem+'.csv')).open('w',newline='') as f:
    fields=[k for k in rows[0] if k not in ('q_rad','q_deg','payload_center_m')]
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
print(json.dumps(eligible,ensure_ascii=False,indent=2))
