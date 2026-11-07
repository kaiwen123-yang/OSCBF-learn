"""Bounded, headless TCP/geometry checks; no ROS, SDK or CAN access."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from piper_oscbf.task_geometry import TaskGeometry, _origin, signed_point_box_distance


def independent_frames(q, width):
    urdf = ET.parse(ROOT/'models/piper_h_with_gripper.urdf').getroot()
    positions = {f'joint{i+1}': float(v) for i, v in enumerate(q)}
    positions['gripper'] = float(width)
    frames = {'world': np.eye(4)}
    pending = list(urdf.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in frames: continue
            name = joint.get('name')
            mimic = joint.find('mimic')
            value = (positions[mimic.get('joint')]*float(mimic.get('multiplier','1'))
                     + float(mimic.get('offset','0')) if mimic is not None else positions.get(name,0.))
            movement = np.eye(4)
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                if joint.get('type') == 'revolute':
                    movement[:3,:3] = Rotation.from_rotvec(axis*value).as_matrix()
                elif joint.get('type') == 'prismatic': movement[:3,3] = axis*value
            frames[child] = frames[parent] @ _origin(joint.find('origin')) @ movement
            pending.remove(joint); progressed = True
        if not progressed: raise ValueError('URDF chain does not resolve')
    return frames


def numeric_jacobian(function, q, eps=1e-6):
    q = np.asarray(q, float)
    return np.stack([(np.asarray(function(q+np.eye(6)[j]*eps))
                      - np.asarray(function(q-np.eye(6)[j]*eps)))/(2*eps) for j in range(6)],axis=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/pick_place.json')
    parser.add_argument('--plan',type=Path,default=ROOT/'results/pick_place_plan_20261104_224921_126900/task.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/task_geometry_check_20261107')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    geom=TaskGeometry(cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    q=np.deg2rad([12.,45.,-70.,18.,35.,-15.])
    maxima={'part_transform_difference':0.,'tcp_transform_difference':0.}
    for width in [0.,.02,.04,.1]:
        frames=independent_frames(q,width)
        actual=np.asarray(geom.part_transforms(q,width))
        for name,tf in zip(geom.part_names,actual):
            maxima['part_transform_difference']=max(maxima['part_transform_difference'],float(np.max(np.abs(tf-frames[name]))))
        tip=np.eye(4);tip[:3,3]=cfg['tcp_offset_m']
        tip[:3,:3]=Rotation.from_euler('xyz',cfg.get('tcp_rpy_rad',[0.,0.,0.])).as_matrix()
        expected=frames['flange_link']@tip
        maxima['tcp_transform_difference']=max(maxima['tcp_transform_difference'],float(np.max(np.abs(expected-np.asarray(geom.tcp_transform(q))))))
    J=np.asarray(geom.tcp_jacobian(q))
    Jp=numeric_jacobian(lambda v:np.asarray(geom.tcp_transform(v))[:3,3],q)
    R=np.asarray(geom.tcp_transform(q))[:3,:3]
    dR=numeric_jacobian(lambda v:np.asarray(geom.tcp_transform(v))[:3,:3],q)
    Jw=[]
    for i in range(6):
        skew=dR[:,:,i]@R.T
        Jw.append(np.array([skew[2,1]-skew[1,2],skew[0,2]-skew[2,0],skew[1,0]-skew[0,1]])*.5)
    maxima['tcp_linear_jacobian_error']=float(np.max(np.abs(J[:3]-Jp)))
    maxima['tcp_angular_jacobian_error']=float(np.max(np.abs(J[3:]-np.stack(Jw,axis=1))))
    # Synthetic obstacle tests are numerical fixtures, never a measured scene.
    fixture=copy.deepcopy(cfg)
    fixture['obstacles']=[{'name':'offline_test_only','center_m':[.43,.07,.18],'size_m':[.05,.06,.12]}]
    fixture.setdefault('runtime_geometry',{})['payload_table_clearance_m']=.005
    fg=TaskGeometry(fixture)
    attachment=np.asarray(fg.attach_cube(q))
    attachment_error=float(np.max(np.abs(np.asarray(fg.cube_world_transform(q,attachment))-fg.world_cube_transform())))
    info=fg.evaluate(q,'transfer',.02,payload_T_tcp=attachment)
    jf=numeric_jacobian(lambda v:np.asarray(fg.barrier_values(v,.02,True,False,attachment,fg.world_cube_transform(),False)),q)
    maxima['barrier_jacobian_error']=float(np.max(np.abs(info['jacobian']-jf)))
    inactive_error=float(np.max(np.abs(info['jacobian'][~info['active_mask']])))
    inactive_value_error=float(np.max(np.abs(info['h'][~info['active_mask']]-1.)))
    table_indices=[i for i,label in enumerate(fg.constraint_labels) if label.startswith('table:payload/')]
    hcontact=np.asarray(fg.barrier_values(q,.02,True,True,attachment))
    hfree=np.asarray(fg.barrier_values(q,.02,True,False,attachment))
    support_gap_error=float(np.max(np.abs(hcontact[table_indices]-hfree[table_indices]-.005)))
    sdf_checks=[]
    for name,point,expected in [('outside',[2.,2.,0.],np.sqrt(2)),('face',[1.,0.,0.],0.),('inside',[0.,0.,0.],-1.)]:
        value=float(signed_point_box_distance(jnp.asarray(point),jnp.ones(3)))
        grad=np.asarray(jax.grad(lambda p:signed_point_box_distance(p,jnp.ones(3)))(jnp.asarray(point)))
        assert np.isfinite(value) and np.all(np.isfinite(grad))
        assert abs(value-expected)<2e-12
        sdf_checks.append({'case':name,'signed_distance_m':value,'gradient':grad.tolist(),
                           'sphere_radius_m':.02,'sphere_clearance_m':value-.02,
                           'physically_clear_for_radius':value-.02>=0,
                           'nonsmooth_note':'Boundary/interior ties provide subgradients; negative clearance is not accepted as a safe initial state.'})
    failures=[]
    for key,value in maxima.items():
        if value>2e-6:failures.append(key)
    assert attachment_error<1e-10 and inactive_error==0 and inactive_value_error==0 and support_gap_error<1e-12
    assert all(m['max_mesh_vertex_outside_cover_m']<=1e-10 for m in geom.mesh_metadata)
    plan=json.loads(args.plan.read_text())
    stage_rows=[]
    gripper_checks=[]
    width=cfg['gripper_open_m']
    payload=np.asarray(geom.default_payload_T_tcp)
    world_cube=geom.world_cube_transform()
    q_last=np.asarray(plan['config']['initial_q_rad'])
    held=False
    h_jit=jax.jit(geom.barrier_values)
    for step in plan['steps']:
        if step['kind']=='gripper':
            name=step['name']
            new_width=cfg['cube_size_m'] if name=='grasp' else cfg['gripper_open_m']
            gripper_checks.append({'stage':name,'post_stage_width_assumption_m':new_width,
                **geom.check_gripper_sweep(q_last,name,width,float(step['width_m']),
                payload_T_tcp=payload,carrying=held,world_cube_T=world_cube)})
            if name=='grasp':
                payload=np.asarray(geom.attach_cube(q_last,world_cube));held=True
            elif name=='release':
                world_cube=np.asarray(geom.cube_world_transform(q_last,payload));held=False
            width=new_width
        elif step['kind']=='arm':
            trajectory=step['trajectory']['joint_trajectory']
            order=[trajectory['joint_names'].index(f'joint{i}') for i in range(1,7)]
            points=np.array([[p['positions'][i] for i in order] for p in trajectory['points']])
            # Every saved point and every segment midpoint; not a continuous-path certificate.
            samples=np.vstack((points,(points[:-1]+points[1:])*.5))
            flags=geom.stage_defaults(step['name'])
            values=np.stack([np.asarray(h_jit(v,width,held,flags['support_contact'],payload,world_cube,
                                               flags['allow_finger_cube_contact'])) for v in samples])
            minimum=np.min(values,axis=0);index=int(np.argmin(minimum))
            stage_rows.append({'stage':step['name'],'samples':len(samples),'carrying':held,
                'width_source':'plan nominal for offline check only','gripper_width_m':width,
                'minimum_h_m':float(minimum[index]),'worst_constraint':geom.constraint_labels[index],
                'negative_samples':int(np.any(values<0,axis=1).sum()),
                'below_minus_1e_6_samples':int(np.any(values < -1e-6,axis=1).sum())})
            q_last=points[-1]
    report={'status':'passed' if not failures else 'failed','numerical_failures':failures,
        'source_config':str(args.config.relative_to(ROOT)),'source_plan':str(args.plan.relative_to(ROOT)),
        'hardware_connected':False,'geometric_model_validation_only':True,
        'max_errors':maxima,'attachment_roundtrip_error':attachment_error,
        'inactive_jacobian_error':inactive_error,'inactive_value_error':inactive_value_error,
        'support_contact_clearance_error':support_gap_error,'sdf_cases':sdf_checks,
        'nominal_path_samples':stage_rows,'gripper_sweeps':gripper_checks,
        'path_scope':'Every saved joint point and each midpoint, nominal grasp widths; not a continuous path or actual measured geometry certification.',
        'fixture_status':'Synthetic obstacle used only for numerical derivative checks; no measured obstacle invented.',
        'metadata':geom.metadata()}
    (args.output/'geometry_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'nominal_stage_geometry.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=stage_rows[0].keys());writer.writeheader();writer.writerows(stage_rows)
    print(json.dumps({k:report[k] for k in ['status','max_errors','attachment_roundtrip_error','nominal_path_samples','gripper_sweeps']},ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
