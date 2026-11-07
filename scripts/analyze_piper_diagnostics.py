"""Offline analysis of recorded Piper runs; no SDK imports or CAN connection."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['pick_place_hardware_20261106_220347_670955',
         'pick_place_hardware_20261106_220924_313468',
         'home_zero_diagnostics_20261106_221059_292266',
         'home_zero_diagnostics_20261106_221142_770204']


def csv_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def percentiles(values):
    return np.percentile(values, [50, 95, 100]).tolist() if values else []


def analyze(name):
    folder = ROOT/'results'/name
    summary = json.loads((folder/'summary.json').read_text())
    recording = json.loads((folder/'recording.json').read_text())
    raw = csv_rows(folder/'raw_can.csv')
    states = [json.loads(line) for line in (folder/'states.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    control = csv_rows(folder/'control.csv') if (folder/'control.csv').exists() else []
    commands = csv_rows(folder/'commands.csv') if (folder/'commands.csv').exists() else []
    errors = [row for row in raw if row['is_error']=='1']
    t0 = float(raw[0]['can_timestamp_s'])
    result = {'directory':str(folder),'error':summary.get('error'),
              'raw_frames':len(raw),'state_snapshots':len(states),
              'control_samples':len(control),'raw_capture_duration_s':float(raw[-1]['can_timestamp_s'])-t0,
              'can_error_frames':len(errors), 'can_error_phases':dict(Counter(r['phase'] for r in errors)),
              'error_decode_source':'/usr/include/linux/can/error.h',
              'recorder_error':recording.get('recorder_error'),
              'driver_feedback_pauses':summary.get('driver_feedback_pauses'),
              'initial_mode':summary.get('initial_status',{}).get('mode_feedback'),
              'final_mode':summary.get('final_mode_feedback',summary.get('final_status',{}).get('mode_feedback')),
              'driver_codes':{},'driver_max_gap_ms':{},'motor_current_range_a':{}}
    for j in range(1,7):
        driver=[r for r in raw if int(r['arbitration_id'],16)==0x260+j and r['dlc']=='8' and r['is_error']=='0']
        result['driver_codes'][str(j)] = dict(Counter(hex(bytes.fromhex(r['data_hex'])[5]) for r in driver))
        result['driver_max_gap_ms'][str(j)] = max(1000*(float(b['can_timestamp_s'])-float(a['can_timestamp_s'])) for a,b in zip(driver,driver[1:]))
        current=[int.from_bytes(bytes.fromhex(r['data_hex'])[2:4],'big',signed=True)*.001
                 for r in raw if int(r['arbitration_id'],16)==0x250+j and r['dlc']=='8' and r['is_error']=='0']
        result['motor_current_range_a'][str(j)] = [min(current),max(current)]
    q=[s['joint']['data'] for s in states if s.get('joint') and s['joint']['timestamp_s']>0]
    result['final_q_deg'] = np.rad2deg(q[-1]).tolist()
    groups=[]
    for row in errors:
        if not groups or float(row['can_timestamp_s'])-float(groups[-1][-1]['can_timestamp_s'])>.5:
            groups.append([])
        groups[-1].append(row)
    bursts=[]
    for group in groups:
        first=group[0]; when=float(first['receive_monotonic_s'])
        prior=[r for r in control if float(r['monotonic_s'])<=when]
        last=prior[-1] if prior else None
        item={'start_elapsed_s':float(first['can_timestamp_s'])-t0,
              'duration_s':float(group[-1]['can_timestamp_s'])-float(first['can_timestamp_s']),
              'frames':len(group),'after_stage':last['stage'] if last else None,
              'protocol_bit1_reports':sum(bool(int(r['arbitration_id'],16)&8 and bytes.fromhex(r['data_hex'])[2]&0x10) for r in group),
              'max_reported_tx_error_counter':max(bytes.fromhex(r['data_hex'])[6] for r in group),
              'has_bus_off':any(int(r['arbitration_id'],16)&0x40 for r in group)}
        if last:
            item['last_control_to_first_error_ms']=1000*(when-float(last['monotonic_s']))
            zeros=[r for r in commands if r['phase']=='zero_velocity' and float(last['monotonic_s'])<float(r['send_start_s'])<=when]
            if zeros:
                zero=zeros[0]
                previous=[r for r in commands if float(r['send_end_s'])<=float(zero['send_start_s'])]
                if previous:
                    item['last_command_end_to_first_zero_start_ms']=1000*(float(zero['send_start_s'])-float(previous[-1]['send_end_s']))
        bursts.append(item)
    result['can_error_bursts']=bursts
    if control:
        result['max_tracking_error_deg']=math.degrees(max(float(r['max_joint_error_rad']) for r in control))
        result['max_command_rad_s']=max(abs(float(r[f'u{j}_sdk_rad_s'])) for r in control for j in range(1,7))
        result['control_cycle_ms_median_p95_max']=percentiles([1000*(float(b['monotonic_s'])-float(a['monotonic_s'])) for a,b in zip(control,control[1:]) if a['stage']==b['stage']])
        result['stage_metrics']={}
        for start in [e for e in events if e['event']=='stage_start']:
            end=next(e for e in events if e['event']=='stage_end' and e['stage']==start['stage'])
            segment=[r for r in control if r['stage']==start['stage']]
            result['stage_metrics'][start['stage']]={'duration_s':end['monotonic_s']-start['monotonic_s'],
                'control_samples':len(segment),'max_tracking_error_deg':math.degrees(max(float(r['max_joint_error_rad']) for r in segment)) if segment else None}
        starts=[e for e in events if e['event']=='stage_start'];ends=[e for e in events if e['event']=='stage_end']
        result['task_duration_s']=ends[-1]['monotonic_s']-starts[0]['monotonic_s']
        grip=csv_rows(folder/'gripper.csv')
        result['gripper_final']={stage:next({k:float(r[k]) for k in ['width_m','force_n','command_width_m','command_force_n']} for r in reversed(grip) if r['stage']==stage) for stage in ['open','grasp','release']}
    if (folder/'sdk_tx.csv').exists():
        tx=csv_rows(folder/'sdk_tx.csv')
        result['sdk_send_calls']=len(tx)
        result['sdk_send_errors']=[r for r in tx if r['exception'] or r['comm_error_after_send']]
        result['sdk_send_id_counts']=dict(Counter(r['arbitration_id'] for r in tx))
    else:
        result['sdk_tx_coverage']='SDK local loopback was off; full raw TX was not captured in this first run. commands.csv has six-axis velocity send records.'
    with (folder/'can_errors.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=raw[0].keys());writer.writeheader();writer.writerows(errors)
    (folder/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result,raw,states


def main():
    analyzed=[analyze(name) for name in NAMES]
    out=ROOT/'results'/NAMES[0]
    comparison={'runs':[v[0] for v in analyzed],
        'operator_environment_confirmation':'User reported no power cycle, cable/power/base/table changes since the previous failed session.',
        'scope':'Hardware records; MoveIt+CPV task and MOVE_J zero hold. No OSCBF task or obstacle demo.',
        'interpretation':'Current errors were not reproduced. CPV runs have CAN error bursts at segment-end zero commands; MOVE_J captures do not. Temporal correlation is not proof that these CAN errors caused the earlier J2 driver fault.'}
    (out/'diagnostic_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for index,(result,raw,states) in enumerate(analyzed):
        col=0 if index<2 else 1
        label=f'Run {index%2+1}'
        first=float(raw[0]['can_timestamp_s'])
        valid=[s for s in states if s.get('joint') and s['joint']['timestamp_s']>0]
        axes[0,col].plot([s['time_s']-first for s in valid],[math.degrees(s['joint']['data'][1]) for s in valid],label=label,lw=1)
        motor=[r for r in raw if r['arbitration_id']=='0x252' and r['dlc']=='8' and r['is_error']=='0']
        axes[1,col].plot([float(r['can_timestamp_s'])-first for r in motor],
                        [int.from_bytes(bytes.fromhex(r['data_hex'])[2:4],'big',signed=True)*.001 for r in motor],label=label,lw=.8)
        if col==0:
            err=[r for r in raw if r['is_error']=='1']
            axes[2,col].scatter([float(r['can_timestamp_s'])-first for r in err],
                                [bytes.fromhex(r['data_hex'])[6] for r in err],label=label,s=9,alpha=.7)
        else:
            driver=[r for r in raw if r['arbitration_id']=='0x262' and r['dlc']=='8' and r['is_error']=='0']
            axes[2,col].plot([float(r['can_timestamp_s'])-first for r in driver],
                            [bytes.fromhex(r['data_hex'])[5] for r in driver],label=label,lw=1)
    axes[0,0].set_title('1.5x complete pick/place — CPV')
    axes[0,1].set_title('MOVE_J zero hold — starts already near zero')
    for col in range(2):
        axes[0,col].set_ylabel('J2 joint feedback [deg]')
        axes[1,col].set_ylabel('J2 motor current [A]')
    axes[2,0].set_ylabel('TX error counter in CAN error frames')
    axes[2,1].set_ylabel('J2 driver status byte')
    axes[2,1].set_yticks([64],['0x40: enabled, no fault'])
    axes[2,1].set_ylim(60,68)
    axes[2,1].text(.5,.7,'0 CAN error frames in both MOVE_J captures',ha='center',transform=axes[2,1].transAxes)
    for ax in axes.flat:
        ax.set_xlabel('Capture elapsed time [s]');ax.grid(alpha=.25);ax.legend(loc='best')
    fig.suptitle('Piper H hardware diagnostics — 2026-11-06; current is SDK-protocol feedback')
    fig.savefig(out/'diagnostic_comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps([{k:v[0].get(k) for k in ['directory','task_duration_s','raw_frames','can_error_frames','max_tracking_error_deg','max_command_rad_s','sdk_send_calls','motor_current_range_a']} for v in analyzed],indent=2))


if __name__=='__main__':
    main()
