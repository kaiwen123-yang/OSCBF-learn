"""Headless candidate search. Candidate coordinates are never field measurements."""
import copy
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import jax
import numpy as np
from piper_oscbf.task_geometry import TaskGeometry

OUT=Path(__file__).resolve().parent
PLAN_NAMES=['pick_place_plan_20261104_224921_126900','pick_place_plan_20261104_232956_473856']
GAPS=[.005,.01,.015,.02,.025,.03,.04,.05]


def sample_plan(task,geom):
    cfg=task['config'];q=np.asarray(cfg['initial_q_rad'],float)
    width=cfg['gripper_open_m'];held=False
    world=geom.world_cube_transform();payload=geom.default_payload_T_tcp.copy()
    rows=[]
    for step in task['steps']:
        if step['kind']=='arm':
            tr=step['trajectory']['joint_trajectory']
            order=[tr['joint_names'].index(f'joint{j}') for j in range(1,7)]
            points=np.array([[p['positions'][i] for i in order] for p in tr['points']])
            original=np.array([p['time_from_start']['sec']+1e-9*p['time_from_start']['nanosec'] for p in tr['points']])
            cap=cfg['joint_velocity_limit_rad_s']*1.5
            durations=np.maximum.reduce([np.diff(original),np.max(np.abs(np.diff(points,axis=0)),axis=1)/(.5*cap),np.full(len(points)-1,.01)])
            knots=np.r_[0,np.cumsum(durations)]
            times=np.unique(np.r_[np.arange(0,knots[-1],.02),knots,knots[-1]])
            samples=np.stack([np.interp(times,knots,points[:,j]) for j in range(6)],axis=1)
            for i,(phase,v) in enumerate(zip(times,samples)):
                rows.append({'stage':step['name'],'kind':'arm','q':v,'width':width,'held':held,
                             'payload':payload.copy(),'endpoint':i in (0,len(samples)-1),'phase_s':float(phase)})
            q=points[-1]
        elif step['kind']=='gripper':
            target=step['width_m']
            for i,w in enumerate(np.linspace(width,target,21)):
                rows.append({'stage':step['name'],'kind':'gripper','q':q.copy(),'width':float(w),'held':held,
                             'payload':payload.copy(),'endpoint':i in (0,20),'phase_s':None})
            if step['name']=='grasp':payload=np.asarray(geom.attach_cube(q,world));held=True;width=cfg['cube_size_m']
            else:
                if step['name']=='release':world=np.asarray(geom.cube_world_transform(q,payload));held=False
                width=cfg['gripper_open_m']
    return rows


def sdf(points,center,half):
    d=np.abs(points-center)-half
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(axis=-1),0)


def main():
    cfg=json.loads((ROOT/'config/pick_place.json').read_text())
    geom=TaskGeometry(cfg)
    half=np.array([.035,.035,.04])
    z=float(cfg['table_top_z_m'])+.04
    finger=np.array([name.split('/')[0] in ('gripper_link1','gripper_link2') for name in geom.sphere_labels])
    plans=[];rows_out=[];detail=[];eligible=[]
    pointmap=jax.jit(jax.vmap(geom.world_geometry,in_axes=(0,0)))
    payloadmap=jax.jit(jax.vmap(geom.cube_world_transform,in_axes=(0,0)))
    for name in PLAN_NAMES:
        task=json.loads((ROOT/'results'/name/'task.json').read_text())
        rows=sample_plan(task,geom)
        q=np.stack([r['q'] for r in rows]);width=np.array([r['width'] for r in rows]);held=np.array([r['held'] for r in rows])
        centers,_=pointmap(q,width);centers=np.asarray(centers)
        # During carrying, cover every finger width in the entire commanded
        # [15,40] mm range; actual runtime still uses measured opening.
        envelope_width=np.where(held,(.015+.04)/2,width)
        envelope_centers,_=pointmap(q,envelope_width);envelope_centers=np.asarray(envelope_centers)
        radii=np.broadcast_to(np.asarray(geom._radii),(len(rows),len(geom.sphere_labels))).copy()
        radii[np.ix_(held,finger)]+=(.04-.015)/4
        payload_tfs=np.asarray(payloadmap(q,np.stack([r['payload'] for r in rows])))
        payload_centers=payload_tfs[:,:3,3]
        stages=np.array([r['stage'] for r in rows]);endpoints=np.array([r['endpoint'] for r in rows])
        working=~np.isin(stages,['open','pregrasp','return_initial'])
        stage_names=list(dict.fromkeys(stages))
        plans.append({'plan':name,'samples':len(rows),'phase_grid_s':.02,'all_saved_knots_included':True,
                      'method':'Original 1.5x path retiming, all saved points and 20 ms reference-phase interpolants; gripper width sweeps included',
                      'physical_obstacle_size_m':[.07,.07,.08],
                      'payload_rotation_bound':'Circumsphere radius sqrt(3)*cube_edge/2, orientation-independent around assumed center',
                      'held_finger_opening_envelope_m':[.015,.04]})
        # Explicit table contact/quantization and orientation-envelope notes.
        contact=[]
        for stage in ('lift','lower'):
            indices=np.flatnonzero(stages==stage)
            i=int(indices[0] if stage=='lift' else indices[-1])
            flags=geom.stage_defaults(stage)
            h=np.asarray(geom.barrier_values(q[i],width[i],True,True,rows[i]['payload']))
            J=np.asarray(geom.barrier_jacobian(q[i],width[i],True,True,rows[i]['payload']))
            subset=np.array([label.startswith('table:payload/') for label in geom.constraint_labels])
            tf=payload_tfs[i];axis=tf[:3,2]
            yaw_support=cfg['cube_size_m']/2*(np.sqrt(2)*np.linalg.norm(axis[:2])+abs(axis[2]))
            arbitrary_sphere=np.sqrt(3)*cfg['cube_size_m']/2
            contact.append({'stage':stage,'min_box_corner_h_m':float(h[subset].min()),
                'minimum_zero_velocity_robust_margin_m_s_alpha3':float((3*h[subset]-.0005*np.abs(J[subset]).sum(axis=1)).min()),
                'minimum_exact_quantized_zero_margin_m_s_alpha3':float((3*h[subset]).min()),
                'upright_initial_arbitrary_yaw_envelope_table_h_m':float(tf[2,3]-cfg['table_top_z_m']-yaw_support),
                'arbitrary_3d_orientation_sphere_table_h_m':float(tf[2,3]-cfg['table_top_z_m']-arbitrary_sphere)})
        plans[-1]['support_contact_checks']=contact
        for x in np.arange(.20,.2801,.01):
            for y in [-.04,-.02,0.,.02,.04]:
                center=np.array([x,y,z])
                robot_nominal=sdf(centers,center,half)-np.asarray(geom._radii)
                robot=sdf(envelope_centers,center,half)-radii
                payload=sdf(payload_centers,center,half)-geom._payload_radius
                physical=np.minimum(robot.min(axis=1),np.where(held,payload,np.inf))
                nominal=np.minimum(robot_nominal.min(axis=1),np.where(held,payload,np.inf))
                five_mm=np.minimum(robot.min(axis=1),np.where(held,payload-.005,np.inf))
                k=int(np.argmin(physical));robot_k=int(np.argmin(robot[k]))
                worst_body='payload' if held[k] and payload[k]<=robot[k,robot_k] else geom.sphere_labels[robot_k]
                minimum=float(physical[k]);goal_min=float(physical[endpoints].min())
                stage_min={s:float(physical[stages==s].min()) for s in stage_names}
                entry={'plan':name,'center_x_m':round(float(x),3),'center_y_m':float(y),'center_z_m':z,
                       'physical_minimum_m':minimum,'nominal_width_physical_minimum_m':float(nominal.min()),
                       'held_center_5mm_uncertainty_minimum_m':float(five_mm.min()),
                       'endpoint_minimum_m':goal_min,'initial_minimum_m':float(physical[0]),'final_minimum_m':float(physical[-1]),
                       'working_subset_minimum_m':float(physical[working].min()),
                       'worst_stage':str(stages[k]),'worst_body':worst_body}
                rows_out.append(entry)
                opportunities=[]
                for gap in GAPS:
                    h=physical-gap
                    active_stages=[s for s in stage_names if np.any(h[stages==s]<0)]
                    opportunity={'virtual_clearance_m':gap,'minimum_virtual_h_m':float(h.min()),
                        'all_stage_endpoints_virtual_feasible':goal_min>=gap,
                        'violation_reference_sample_fraction':float(np.mean(h<0)),
                        'activation_candidate_stages':active_stages,
                        'qualified':bool(minimum>=.005 and goal_min>=gap+.002 and np.min(h)<-.002)}
                    opportunities.append(opportunity)
                    if opportunity['qualified']:eligible.append({**entry,**opportunity})
                detail.append({**entry,'stage_minimum_physical_clearance_m':stage_min,'virtual_gap_cases':opportunities})
    report={'status':'offline_candidate_search_complete','physical_hardware_access':False,
        'obstacle_observation':{'shape':'upright cylinder','diameter_m':.07,'height_m':.08,
            'dimensions_status':'operator_approximate','xy_status':'unmeasured: roughly 20+ cm is not a base_link coordinate',
            'box_envelope_m':[.07,.07,.08],'center_z_assumes_resting_on_table_m':z},
        'plans':plans,'candidates':detail,'qualified_candidates':eligible,
        'limitations':['All distances are model calculations with conservative sphere coverage, not metrology.',
            'Swept-path sampling is not a continuous collision proof.',
            'The cylinder dimensions and position need field confirmation before hardware execution.',
            'Held cube center uncertainty of 5 mm is a sensitivity case, not an operator-verified bound.',
            'Uniform virtual clearance must remain feasible at every required stage endpoint.',
            'No candidate is a hardware-ready plan until MoveIt replan and filtered simulation validate it.']}
    (OUT/'candidate_search.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'candidate_clearances.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows_out[0].keys());writer.writeheader();writer.writerows(rows_out)
    print(json.dumps({'plans':plans,'qualified_candidates':eligible,
        'central_candidates':[r for r in rows_out if r['center_y_m']==0.]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
