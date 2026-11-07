"""Essential static lifecycle/logging checks. No simulation loop, SDK or CAN."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from piper_oscbf.task_filter import TaskFilterError
from piper_oscbf.task_reference import PhaseGovernor, StageReference
from piper_oscbf.task_session import TaskSession, jsonable


def main():
    source=ROOT/'results/pick_place_plan_20261107_193629_186524/task.json'
    task=json.loads(source.read_text())
    out=ROOT/'results/task_session_check_20261107'
    out.mkdir(exist_ok=True)
    session=TaskSession(task,'oscbf')
    q=np.asarray(task['config']['initial_q_rad'])
    cap=task['config']['joint_velocity_limit_rad_s']
    result=session.filter_step(q,q,np.zeros(6),.04,cap,'pregrasp')
    control_row={'stage':'pregrasp','monotonic_s':0.}
    session.add_control(control_row,result)
    hold=session.validate_hold(q,.04,cap,'open')
    session.add_hold({'stage':'open','monotonic_s':0.},hold)
    arms={s['name']:s for s in task['steps'] if s['kind']=='arm'}
    endpoint=lambda name:np.asarray(arms[name]['trajectory']['joint_trajectory']['points'][-1]['positions'])
    grasp_q,release_q=endpoint('approach'),endpoint('lower')
    grasp_sweep=session.check_gripper(grasp_q,.04,.015,'grasp')
    original_world=session.world_cube_T.copy()
    grasp=session.after_gripper('grasp',grasp_q,.02)
    assert session.carrying and grasp['carrying_after']
    assert np.allclose(np.asarray(session.geometry.cube_world_transform(grasp_q,session.payload_T_tcp)),original_world,atol=1e-12)
    contact_hold=session.validate_hold(grasp_q,.02,cap,'grasp')
    session.add_hold({'stage':'grasp','monotonic_s':1.},contact_hold)
    release_sweep=session.check_gripper(release_q,.02,.04,'release')
    expected_world=np.asarray(session.geometry.cube_world_transform(release_q,session.payload_T_tcp))
    release=session.after_gripper('release',release_q,.04)
    assert not session.carrying and np.array_equal(session.world_cube_T,expected_world)
    reference=StageReference(arms['pregrasp'],cap)
    governor=PhaseGovernor(reference,task['config']['arrival_tolerance_rad'],**session.governor_options)
    state=governor.update(q,now=0.)
    session.add_schedule('pregrasp',reference,[state.as_dict()])
    with (out/'control.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=control_row);writer.writeheader();writer.writerow(control_row)
    summary=session.finish(out)
    assert summary['control_records']==1 and summary['hold_records']==2
    assert summary['control_alignment']['status']=='matched'
    assert set(summary['fingerprints'])=={'path','geometry','parameters','controller','constraints','model'}
    with np.load(out/'filter_constraints.npz',allow_pickle=False) as archive:
        assert archive['h'].shape==(1,len(session.constraint_labels))
        assert archive['control_row_index'].tolist()==[0]
    holds=[json.loads(line) for line in (out/'hold_safety.jsonl').read_text().splitlines()]
    assert [h['hold_row_index'] for h in holds]==[0,1]
    assert all(h['q'] is not None and h['gripper_width_m'] is not None for h in holds)
    # A single focused nonzero-hold rejection check; no dynamics or device calls.
    guard=TaskSession.__new__(TaskSession)
    guard.failures=[]
    guard.filter_step=lambda *_: {'u_sdk':np.array([.001,0,0,0,0,0])}
    try:
        guard.validate_hold(q,.04,cap,'open')
    except TaskFilterError as exc:
        assert exc.reason=='hold_requires_nonzero_command'
    else:
        raise AssertionError('Nonzero hold request was accepted')
    encoded=jsonable({'values':np.array([np.nan,np.inf,-np.inf]),'flag':np.bool_(True)})
    json.dumps(encoded,allow_nan=False)
    assert encoded['values']==['NaN','Infinity','-Infinity']
    report={'status':'passed','scope':'Static filter/session lifecycle and artifact alignment; no closed-loop simulation or hardware',
        'source_plan':str(source.relative_to(ROOT)),'hardware_connected':False,
        'initial_hold':session.initial_hold,'grasp_sweep':jsonable(grasp_sweep),
        'release_sweep':jsonable(release_sweep),'nonzero_hold_rejected':True,
        'strict_json_nonfinite_preserved':True,'finish_summary':summary}
    (out/'check.json').write_text(json.dumps(jsonable(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','initial_hold':session.initial_hold,
        'grasp_min_h_m':grasp_sweep['minimum_h_lower_bound_m'],
        'release_min_h_m':release_sweep['minimum_h_lower_bound_m'],
        'artifact_alignment':summary['control_alignment'],'nonzero_hold_rejected':True},indent=2))


if __name__=='__main__':main()
