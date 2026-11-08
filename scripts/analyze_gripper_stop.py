"""Read-only reconstruction of the recorded gripper pose-guard stop; no plots/CAN."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FAILED='pick_place_hardware_20261107_195308_282501'
BASELINES=['pick_place_hardware_20261106_220347_670955',
           'pick_place_hardware_20261106_220924_313468',
           'pick_place_hardware_20261107_180811_947624',
           'pick_place_hardware_20261107_190744_717616']

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def drift(run):
    p=ROOT/'results'/run
    events=lines(p/'events.jsonl')
    start=next(e for e in events if e['event']=='stage_start' and e.get('stage')=='grasp')
    ends=[e for e in events if (e['event']=='stage_end' and e.get('stage')=='grasp')
          or e['event']=='exception_before_stop']
    end=next(e for e in ends if e['monotonic_s']>=start['monotonic_s'])
    anchor=np.array(start['state']['joint']['data'])
    data=[r for r in rows(p/'feedback.csv') if start['monotonic_s']<=float(r['monotonic_s'])<=end['monotonic_s']]
    q=np.array([[float(r[f'q{j}_rad']) for j in range(1,7)] for r in data])
    deviation=np.abs(q-anchor)
    peak=np.unravel_index(deviation.argmax(),deviation.shape)
    summary=json.loads((p/'summary.json').read_text())
    return {'run':run,'status':summary['status'],'object_transfer_success':summary.get('object_transfer_success'),
        'anchor_source':'stage_start SDK snapshot; not a separately recorded local Python anchor',
        'anchor_q_rad':anchor.tolist(),'start_monotonic_s':start['monotonic_s'],
        'duration_s':end['monotonic_s']-start['monotonic_s'],'feedback_samples':len(data),
        'max_delta_rad':float(deviation[peak]),'max_delta_deg':float(np.rad2deg(deviation[peak])),
        'max_joint':int(peak[1]+1),'samples_over_0_002rad':int((deviation.max(axis=1)>.002).sum()),
        'first_over_elapsed_s':next((float(r['monotonic_s'])-start['monotonic_s']
            for r,v in zip(data,deviation.max(axis=1)) if v>.002),None)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/gripper_stop_diagnosis_20261108')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    p=ROOT/'results'/FAILED
    summary=json.loads((p/'summary.json').read_text());events=lines(p/'events.jsonl')
    holds=lines(p/'hold_safety.jsonl');feedback=rows(p/'feedback.csv');tx=rows(p/'sdk_tx.csv');raw=rows(p/'raw_can.csv')
    start=next(e for e in events if e['event']=='stage_start' and e.get('stage')=='grasp')
    stop=next(e for e in events if e['event']=='exception_before_stop')
    first_hold=next(h for h in holds if h['stage']=='grasp')
    anchor=np.array(start['state']['joint']['data']); stopped=np.array(stop['state']['joint']['data'])
    command=next(r for r in tx if r['phase']=='grasp' and r['arbitration_id']=='0x159')
    command_time=float(command['send_start_s'])
    prior=[r for r in feedback if start['monotonic_s']-.5<=float(r['monotonic_s'])<=start['monotonic_s']]
    window=[r for r in feedback if command_time-.1<=float(r['monotonic_s'])<=stop['monotonic_s']+.2]
    timeline=[]
    for r in window:
        q=np.array([float(r[f'q{j}_rad']) for j in range(1,7)])
        delta=q-anchor
        timeline.append({'relative_to_close_s':float(r['monotonic_s'])-command_time,
            'monotonic_s':float(r['monotonic_s']),'can_timestamp_s':float(r['can_timestamp_s']),
            'j5_rad':q[4],'j5_delta_rad':delta[4],'j5_delta_deg':math.degrees(delta[4]),
            'max_delta_rad':float(np.max(np.abs(delta))),
            'guard_exceeded':bool(np.max(np.abs(delta))>.002),
            'motor5_position_rad':r['motor5_position_rad'],
            'motor5_velocity_rad_s':r['motor5_velocity_rad_s']})
    selected=[r for r in raw if command_time-.1<=float(r['receive_monotonic_s'])<=stop['monotonic_s']+.2
              and r['arbitration_id'] in ('0x2a7','0x255','0x265','0x2a8')]
    j5frames=[]
    for r in selected:
        if r['arbitration_id']=='0x2a7':
            value=int.from_bytes(bytes.fromhex(r['data_hex'])[:4],'big',signed=True)/1000
            j5frames.append({'timestamp_s':float(r['can_timestamp_s']), 'j5_deg':value,'data_hex':r['data_hex']})
    matched=[r for r in j5frames if abs(r['j5_deg']-math.degrees(stopped[4]))<1e-8
             and r['timestamp_s']<=stop['time_s']]
    raw_errors=[r for r in raw if r['is_error']=='1']
    controls=rows(p/'control.csv')
    drivers={str(j):dict(Counter(hex(bytes.fromhex(r['data_hex'])[5]) for r in raw
        if r['is_error']=='0' and int(r['arbitration_id'],16)==0x260+j and r['dlc']=='8')) for j in range(1,7)}
    result={'scope':'Recorded hardware data and source inspection, no new hardware operations',
        'run':FAILED,'source_commit':json.loads((p/'recording.json').read_text())['git_commit'],
        'exception':summary['error'],'completed_stages':summary['completed_stages'],
        'stop_action':summary['stop_action'],'anchor_q_rad':anchor.tolist(),
        'anchor_evidence':'grasp stage_start and first grasp hold contain identical six-axis values',
        'stopped_q_rad':stopped.tolist(),'delta_rad':(stopped-anchor).tolist(),
        'delta_deg':np.rad2deg(stopped-anchor).tolist(),
        'guard_rad':.002,'guard_deg':math.degrees(.002),
        'stage_start_to_exception_ms':1000*(stop['monotonic_s']-start['monotonic_s']),
        'close_send_to_exception_ms':1000*(summary['exception_monotonic_s']-command_time),
        'feedback_age_at_exception_ms':1000*(stop['time_s']-stop['state']['joint']['timestamp_s']),
        'prior_half_second_joint_range_deg':[
            math.degrees(max(float(r[f'q{j}_rad']) for r in prior)-min(float(r[f'q{j}_rad']) for r in prior))
            for j in range(1,7)],
        'first_close_command':command,'raw_j5_confirmation':matched[-1],
        'can_error_frames':len(raw_errors),'driver_codes':drivers,
        'sdk_send_errors':sum(bool(r['exception'] or r['comm_error_after_send']) for r in tx),
        'qp_solved_control_samples':sum(r['qp_solved']=='True' for r in controls),
        'filter_changed_control_samples':sum(r['constraint_active']=='True' for r in controls),
        'all_logged_grasp_arm_requests_zero':all(all(float(r[f'u{j}_rad_s'])==0 for j in range(1,7))
             for r in rows(p/'commands.csv') if r['phase']=='grasp'),
        'historical_grasp_drift':[drift(name) for name in BASELINES]+[drift(FAILED)],
        'evidence_limits':['No external position measurement or video was available.',
            'Temporal association with closing does not distinguish mechanical coupling, contact, actuator behavior or firmware feedback behavior.',
            'Recorded local Python anchor was not captured directly; matching stage_start and first-hold snapshots constrain it.'],
        'input_sha256':{f:sha(p/f) for f in ['summary.json','events.jsonl','feedback.csv','raw_can.csv',
            'sdk_tx.csv','hold_safety.jsonl','control.csv','sources/task_execution.py','sources/task_session.py']}}
    for filename,data in [('feedback_window.csv',timeline),('raw_packet_window.csv',selected),('raw_j5_decoded.csv',j5frames)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['delta_deg','guard_deg','close_send_to_exception_ms',
        'feedback_age_at_exception_ms','prior_half_second_joint_range_deg','can_error_frames',
        'qp_solved_control_samples','historical_grasp_drift']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
