"""Read-only numerical analysis of the interrupted OSCBF hardware recording."""
import csv
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

p = Path(__file__).resolve().parent
s = json.loads((p/'summary.json').read_text())
r = list(csv.DictReader((p/'raw_can.csv').open()))
tx = list(csv.DictReader((p/'sdk_tx.csv').open()))
c = list(csv.DictReader((p/'control.csv').open()))
holds = [json.loads(x) for x in (p/'hold_safety.jsonl').read_text().splitlines()]
events = [json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
f = np.load(p/'filter_constraints.npz')
before = json.loads((p/'can_interface_before.json').read_text())[0]
after = json.loads((p/'can_interface_after.json').read_text())[0]
numbers = lambda key: np.array([float(x[key]) for x in c])
active = np.array([x['constraint_active']=='True' for x in c])
solved = np.array([x['qp_solved']=='True' for x in c])
converged = np.array([x['qp_converged']=='True' for x in c])
times, h = numbers('monotonic_s'), numbers('min_h_m')
weights = np.r_[np.diff(times), 0.]
for i in range(len(c)-1):
    if c[i]['stage'] != c[i+1]['stage']:
        weights[i] = 0.
bad = [x for x in tx if x['exception'] or x['comm_error_after_send']]
normal = [x for x in r if x['is_rx']=='1' and x['is_error']=='0']
gcount = int(f['geometry_constraint_count'])
mask = f['constraint_mask'][:, :gcount].astype(bool)
physical = np.where(mask, f['physical_h'][:, :gcount], np.inf)
minimum = np.unravel_index(np.argmin(physical), physical.shape)
local = lambda epoch: datetime.fromtimestamp(float(epoch), ZoneInfo('Asia/Shanghai')).isoformat()
distribution = lambda a: {'median':float(np.median(a)), 'p95':float(np.quantile(a,.95)), 'maximum':float(np.max(a))}
stages = []
for start in (e for e in events if e['event']=='stage_start'):
    end = next((e for e in events if e['event']=='stage_end' and e['stage']==start['stage']), None)
    stages.append({'stage':start['stage'], 'completed':end is not None,
        'observed_duration_s':(end['monotonic_s'] if end else s['exception_monotonic_s'])-start['monotonic_s']})
negative = [{'control_row_index':i,'stage':c[i]['stage'],'monotonic_s':times[i],
             'min_h_m':h[i],'until_next_same_stage_sample_s':weights[i]} for i in range(len(c)) if h[i]<0]
report = {
    'status':s['status'],'error':s['error'],'completed_stages':s['completed_stages'],
    'operator_confirmed_object_on_left':True,'operator_reported_arm_disabled':True,
    'controller_disable_time_observed':False,'full_sequence_completed':False,
    'stages':stages,'control_records':len(c),'filter_active_samples':int(active.sum()),
    'filter_active_sample_fraction':float(active.mean()),
    'filter_active_observed_seconds':float(weights[active].sum()),
    'qp_calls':int(solved.sum()),'qp_nonconverged_calls':int((solved & ~converged).sum()),
    'filter_correction_l2_max_rad_s':float(numbers('filter_correction_joint_l2_rad_s').max()),
    'tcp_error_rms_mm':float(np.sqrt(np.mean(numbers('tcp_position_error_m')**2))*1000),
    'tcp_error_max_mm':float(numbers('tcp_position_error_m').max()*1000),
    'active_tcp_error_rms_mm':float(np.sqrt(np.mean(numbers('tcp_position_error_m')[active]**2))*1000),
    'active_orientation_error_rms_deg':float(np.rad2deg(np.sqrt(np.mean(numbers('orientation_error_rad')[active]**2)))),
    'minimum_model_h_m':float(h.min()),'minimum_constraint_label':str(f['constraint_labels'][minimum[1]]),
    'negative_samples':negative,'negative_sample_count':len(negative),
    'negative_below_1nm_sample_count':int((h < -1e-9).sum()),
    'negative_observed_seconds':float(weights[h<0].sum()),
    'negative_below_1nm_observed_seconds':float(weights[h < -1e-9].sum()),
    'negative_duration_convention':'Left sample held until next sample in same stage, excluding unobserved interstage intervals; not external physical contact duration.',
    'hold_minimum_model_h_m':min(x['min_h_m'] for x in holds),
    'hold_negative_samples':sum(x['min_h_m']<0 for x in holds),
    'minimum_applied_margin':float(numbers('min_applied_margin').min()),
    'control_cycle_ms':distribution(numbers('phase_dt_s')[numbers('phase_dt_s')>0]*1000),
    'feedback_callback_age_ms':distribution(numbers('feedback_age_s')*1000),
    'qp_solve_ms':distribution(numbers('qp_ms')[solved]),
    'raw_frames':len(r),'can_error_frames':sum(x['is_error']=='1' for x in r),
    'can_error_counter_delta':{k:after['linkinfo']['info_xstats'][k]-v for k,v in before['linkinfo']['info_xstats'].items()},
    'sdk_send_calls':len(tx),'sdk_error_calls':len(bad),'first_sdk_error':bad[0],
    'last_normal_receive':normal[-1],'last_normal_receive_local':local(normal[-1]['can_timestamp_s']),
    'exception_local':local(s['exception_time_s']),
    'first_sdk_error_local':local(bad[0]['time_s']),
    'usbip_disconnect_monotonic_s':1761.230416,
    'receive_end_to_usbip_disconnect_s':1761.230416-float(normal[-1]['receive_monotonic_s']),
    'usbip_windows_event_local':'2026-11-08T22:19:13+08:00',
    'operator_usb_or_wsl_intervention':'没有拔插，也没有关闭 WSL',
    'wsl_keepalive_observation':'PID568 bash --noprofile --norc -i remained alive; no matching VM restart in kernel log',
    'interpretation':'USB data/transfer path stalled and later disconnected without operator intervention. This does not isolate WSL kernel vs Windows USB/IP vs adapter/USB hardware, nor explain earlier CAN ACK errors.'}
(p/'result_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('negative_samples','stages','first_sdk_error','last_normal_receive')},ensure_ascii=False,indent=2))
