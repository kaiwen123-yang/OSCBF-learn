"""Text/JSON-only inspection of a recorded complete or interrupted CPV task."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def stats(values):
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    return dict(zip(('min', 'median', 'p95', 'max'), map(float, np.percentile(data, [0, 50, 95, 100]))))


def analyze(folder):
    summary = json.loads((folder/'summary.json').read_text())
    recording = json.loads((folder/'recording.json').read_text())
    commands, control, feedback, raw, tx = [csv_rows(folder/f'{key}.csv')
        for key in ('commands','control','feedback','raw_can','sdk_tx')]
    normal_intervals, transitions = [], []
    for previous, current in zip(commands, commands[1:]):
        immediate = current.get('pacing_bypassed', '').lower() in ('true','1')
        interval = float(current['send_start_s'])-float(previous['send_start_s'])
        if not immediate:
            normal_intervals.append(interval*1000)
        if current['phase']=='zero_velocity' and previous['phase']!='zero_velocity':
            transitions.append({'previous_phase':previous['phase'],
                'start_interval_ms':interval*1000,
                'previous_end_to_zero_start_ms':1000*(float(current['send_start_s'])-float(previous['send_end_s'])),
                'immediate':immediate})
    errors = [r for r in raw if r['is_error']=='1']
    drivers = {str(j):dict(Counter(hex(bytes.fromhex(r['data_hex'])[5]) for r in raw
                    if r['is_error']=='0' and int(r['arbitration_id'],16)==0x260+j and r['dlc']=='8')) for j in range(1,7)}
    cycles = [1000*(float(b['monotonic_s'])-float(a['monotonic_s']))
              for a,b in zip(control,control[1:]) if a['stage']==b['stage']]
    feedback_gaps = [1000*(float(b['can_timestamp_s'])-float(a['can_timestamp_s']))
                     for a,b in zip(feedback,feedback[1:])]
    ages = [1000*float(r['feedback_age_s']) for r in control if r.get('feedback_age_s')]
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    starts = [e for e in events if e['event']=='stage_start']
    ends = [e for e in events if e['event']=='stage_end']
    period = summary.get('cpv_period_s')
    result = {'directory':str(folder.resolve()), 'status':summary['status'], 'error':summary.get('error'),
        'git_commit':recording.get('git_commit'), 'tracked_source_changes':recording.get('tracked_source_changes'),
        'completed_stages':summary.get('completed_stages'),
        'object_transfer_success':summary.get('object_transfer_success'),
        'speed_scale':summary.get('speed_scale'), 'descent_speed_scale':summary.get('descent_speed_scale'),
        'task_duration_s':ends[-1]['monotonic_s']-starts[0]['monotonic_s'] if starts and ends else None,
        'pacing_period_s':period, 'normal_start_intervals_ms':stats(normal_intervals),
        'normal_short_interval_count':sum(v < period*1000-.001 for v in normal_intervals) if period else None,
        'normal_zero_transitions':transitions, 'can_error_frames':len(errors),
        'can_error_phases':dict(Counter(r['phase'] for r in errors)),
        'bus_off_error_frames':sum(bool(int(r['arbitration_id'],16)&0x40) for r in errors),
        'control_cycle_ms':stats(cycles), 'feedback_gap_ms':stats(feedback_gaps),
        'feedback_age_ms':stats(ages),
        'max_tracking_error_deg':math.degrees(max(float(r['max_joint_error_rad']) for r in control)) if control else None,
        'driver_codes':drivers, 'driver_feedback_pauses':summary.get('driver_feedback_pauses'),
        'sdk_send_calls':len(tx), 'sdk_send_error_count':sum(bool(r['exception'] or r['comm_error_after_send']) for r in tx),
        'recorder_error':recording.get('recorder_error'), 'interface_delta':recording.get('interface_delta'),
        'timing_scope':'SDK call-group monotonic times, not physical CAN wire timestamps'}
    result['communication_checks_passed'] = bool(summary['status']=='sequence_completed'
        and period and normal_intervals and result['normal_short_interval_count']==0
        and not errors and not result['sdk_send_error_count'] and not result['recorder_error']
        and not result['driver_feedback_pauses'] and all(set(codes)=={'0x40'} for codes in drivers.values()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--baseline', type=Path)
    args=parser.parse_args()
    result=analyze(args.result)
    report={'run':result,'baseline':analyze(args.baseline) if args.baseline else None}
    (args.result/'cpv_timing_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
