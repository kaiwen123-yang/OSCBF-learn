"""Read-only comparison of the tracking peak after CPV pacing repair.

Writes text/CSV/JSON only into this review directory. Does not import any
hardware entry, SDK or plotting library.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def stat(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'median': float(np.median(a)), 'p95': float(np.percentile(a,95)),
            'p99': float(np.percentile(a,99)), 'max': float(np.max(a)),
            'rms': float(np.sqrt(np.mean(a*a)))}


def decoded(row):
    aid = int(row['arbitration_id'],16)
    data = bytes.fromhex(row['data_hex'])
    if row['is_error'] != '0' or len(data) != 8: return None
    item = {'can_timestamp_s':float(row['can_timestamp_s']),
            'receive_monotonic_s':float(row['receive_monotonic_s']),
            'phase':row['phase'], 'arbitration_id':row['arbitration_id'], 'data_hex':row['data_hex']}
    if aid in (0x2a5,0x2a6,0x2a7):
        first = (aid-0x2a5)*2+1
        item.update(kind='joint', first_joint=first,
                    value1_rad=int.from_bytes(data[:4],'big',signed=True)*.001*math.pi/180,
                    value2_rad=int.from_bytes(data[4:],'big',signed=True)*.001*math.pi/180)
    elif 0x251 <= aid <= 0x256:
        item.update(kind='motor', joint=aid-0x250,
                    velocity_rad_s=int.from_bytes(data[:2],'big',signed=True)*.001,
                    current_a=int.from_bytes(data[2:4],'big',signed=True)*.001,
                    position_rad=int.from_bytes(data[4:],'big',signed=True)*.001)
    else: return None
    return item


def analyze(path, label):
    control = read_csv(path/'control.csv')
    feedback = read_csv(path/'feedback.csv')
    commands = read_csv(path/'commands.csv')
    tx = read_csv(path/'sdk_tx.csv')
    raw = read_csv(path/'raw_can.csv')
    events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    summary = json.loads((path/'summary.json').read_text())
    peak_index = max(range(len(control)),key=lambda k:float(control[k]['max_joint_error_rad']))
    peak = control[peak_index]
    peak_time = float(peak['monotonic_s'])
    stage_data = {}
    for stage in dict.fromkeys(r['stage'] for r in control):
        c = [r for r in control if r['stage']==stage]
        errors = np.asarray([[float(r[f'q{j}_ref_rad'])-float(r[f'q{j}_rad']) for j in range(1,7)] for r in c])
        maximum = np.rad2deg(np.max(np.abs(errors),axis=1))
        t = np.asarray([float(r['monotonic_s']) for r in c])
        gaps = np.diff(t)
        weights = np.r_[gaps, np.median(gaps)] if len(c)>1 else np.ones(1)
        exceed = np.max(np.abs(errors),axis=1) > .01
        local_peak = int(np.argmax(maximum))
        start_event = next(e for e in events if e['event']=='stage_start' and e['stage']==stage)
        end_event = next(e for e in events if e['event']=='stage_end' and e['stage']==stage)
        m5 = [v for r in raw if r['phase']==stage and int(r['arbitration_id'],16)==0x255
              for v in [decoded(r)] if v]
        stage_data[stage] = {
            'max_joint_error_deg_stats':stat(maximum),
            'time_weighted_rms_max_joint_error_deg':float(np.sqrt(np.average(maximum**2,weights=weights))),
            'per_joint_max_abs_error_deg':np.rad2deg(np.max(np.abs(errors),axis=0)).tolist(),
            'peak_joint':int(np.argmax(np.abs(errors[local_peak])))+1,
            'duration_s':end_event['monotonic_s']-start_event['monotonic_s'],
            'control_rows_above_001rad':int(exceed.sum()),
            'estimated_time_above_001rad_s':float(weights[exceed].sum()),
            'zero_reference_rate_rows':sum(float(r['progress_rate'])==0 for r in c),
            'motor5_reported_velocity_range_rad_s':[min(v['velocity_rad_s'] for v in m5),max(v['velocity_rad_s'] for v in m5)],
            'motor5_frames_abs_velocity_above_05':sum(abs(v['velocity_rad_s'])>.5 for v in m5),
            'motor5_frames':len(m5),
        }
    all_errors = np.rad2deg([float(r['max_joint_error_rad']) for r in control])
    outside = np.rad2deg([float(r['max_joint_error_rad']) for r in control if r['stage']!='approach'])
    global_peak_joint = max(range(1,7),key=lambda j:abs(float(peak[f'q{j}_ref_rad'])-float(peak[f'q{j}_rad'])))
    peak_feedback = next(r for r in feedback if r['sequence']==peak['sample_sequence'])
    peak_start = float(peak['can_timestamp_s'])
    q_packet = min([v for r in raw for v in [decoded(r)]
                    if v and v['kind']=='joint' and v['first_joint']==5
                    and abs(v['value1_rad']-float(peak['q5_rad']))<1e-12
                    and abs(v['can_timestamp_s']-peak_start)<.02],
                   key=lambda v:abs(v['can_timestamp_s']-peak_start))
    motor_time = float(peak_feedback['motor5_timestamp_s'])
    motor_packet = min([v for r in raw if int(r['arbitration_id'],16)==0x255
                        for v in [decoded(r)] if v],key=lambda v:abs(v['can_timestamp_s']-motor_time))
    assert abs(motor_packet['can_timestamp_s']-motor_time)<1e-6
    assert abs(motor_packet['position_rad']-float(peak_feedback['motor5_position_rad']))<1e-12
    assert abs(motor_packet['velocity_rad_s']-float(peak_feedback['motor5_velocity_rad_s']))<1e-12
    nearby_controls = [r for r in control if abs(float(r['monotonic_s'])-peak_time)<=.35]
    nearby_feedback = [r for r in feedback if abs(float(r['monotonic_s'])-peak_time)<=.35]
    nearby_commands = [r for r in commands if abs(float(r['send_start_s'])-peak_time)<=.35]
    nearby_raw = [r for r in raw if abs(float(r['receive_monotonic_s'])-peak_time)<=.35
                  and int(r['arbitration_id'],16) in (0x2a5,0x2a6,0x2a7,0x254,0x255,0x256,0x264,0x265,0x266)]
    decoded_nearby = [v for r in nearby_raw for v in [decoded(r)] if v]
    decoded_flat = []
    for v in decoded_nearby:
        decoded_flat.append({k:v.get(k) for k in ('can_timestamp_s','receive_monotonic_s','phase','arbitration_id',
                            'data_hex','kind','first_joint','joint','value1_rad','value2_rad',
                            'velocity_rad_s','current_a','position_rad')})
    window_packets = {}
    for aid in (0x2a7,0x254,0x255,0x256):
        d = [v for v in decoded_nearby if int(v['arbitration_id'],16)==aid]
        times = np.array([v['can_timestamp_s'] for v in d])
        item = {'frames':len(d), 'interval_ms':stat(np.diff(times)*1000)}
        if aid==0x2a7:
            pos = np.array([v['value1_rad'] for v in d])
            item.update(joint5_reported_position_range_rad=[float(pos.min()),float(pos.max())],
                        joint5_position_increment_abs_max_rad=float(np.max(np.abs(np.diff(pos)))),
                        apparent_joint5_finite_difference_velocity_range_rad_s=[float(np.min(np.diff(pos)/np.diff(times))),float(np.max(np.diff(pos)/np.diff(times)))])
        else:
            item.update(reported_position_range_rad=[min(v['position_rad'] for v in d),max(v['position_rad'] for v in d)],
                        reported_velocity_range_rad_s=[min(v['velocity_rad_s'] for v in d),max(v['velocity_rad_s'] for v in d)])
        window_packets[hex(aid)] = item
    top_tx = max(tx,key=lambda r:float(r['send_end_s'])-float(r['send_start_s']))
    near_tx = [r for r in tx if abs(float(r['send_start_s'])-peak_time)<=.35]
    connected = next(e for e in events if e['event']=='connected')['state']
    packet_sources = {'joint5_peak_raw_packet':q_packet, 'motor5_peak_cached_raw_packet':motor_packet}
    result = {'name':path.name, 'source_directory':str(path), 'status':summary['status'],
        'initial_mode':connected['arm']['data']['mode_feedback'],
        'initial_q_rad':connected['joint']['data'],
        'driver_feedback_wait_events':summary['driver_feedback_pauses'],
        'overall_error_deg':stat(all_errors), 'without_approach_error_deg':stat(outside),
        'stages':stage_data, 'peak':{'control_index':peak_index,'joint':global_peak_joint,
            'control_row':peak,'feedback_row':peak_feedback,**packet_sources,
            'raw_packet_window':window_packets,
            'window_max_sdk_send_duration_ms':max((float(r['send_end_s'])-float(r['send_start_s']))*1000 for r in near_tx),
            'window_max_command_group_interval_ms':max((float(b['send_start_s'])-float(a['send_start_s']))*1000
                    for a,b in zip(nearby_commands,nearby_commands[1:])),
            'window_reference_q5_range_rad':[min(float(r['q5_ref_rad']) for r in nearby_controls),max(float(r['q5_ref_rad']) for r in nearby_controls)],
            'window_requested_q5_velocity_range_rad_s':[min(float(r['u5_sdk_rad_s']) for r in nearby_controls),max(float(r['u5_sdk_rad_s']) for r in nearby_controls)]},
        'largest_sdk_send_call':{**top_tx,'duration_ms':(float(top_tx['send_end_s'])-float(top_tx['send_start_s']))*1000,
                                'seconds_before_peak':peak_time-float(top_tx['send_start_s'])},
        'source_sha256':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ['control.csv','feedback.csv','raw_can.csv','sdk_tx.csv']}}
    save_csv(OUT/f'{label}_peak_control.csv',nearby_controls)
    save_csv(OUT/f'{label}_peak_feedback.csv',nearby_feedback)
    save_csv(OUT/f'{label}_peak_commands.csv',nearby_commands)
    save_csv(OUT/f'{label}_peak_raw.csv',nearby_raw)
    save_csv(OUT/f'{label}_peak_raw_decoded.csv',decoded_flat)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/pick_place_hardware_20261107_180811_947624')
    parser.add_argument('--paced',type=Path,default=Path('/home/kaiwen/OSCBF-cpv-validation/results/pick_place_hardware_20261107_190744_717616'))
    args = parser.parse_args()
    baseline, paced = analyze(args.baseline,'baseline'), analyze(args.paced,'paced')
    report = {'scope':'Offline comparison of recorded feedback and command calls; no hardware/control change',
        'baseline':baseline,'paced':paced,
        'units_source':'Fixed pyAgxArm commit d90e234ba8e408cfa964b38aa9120b3131fc9f85, protocols/can_protocol/drivers/piper/default/parser.py: joint packets int32*0.001 degree→rad; motor velocity int16*0.001 rad/s, motor position int32*0.001 rad.',
        'interpretation_limits':['Both feedback families originate from robot protocol messages, not external motion capture.',
            'Motor fields are retained as decoded by the fixed SDK; no unverified gear-ratio conversion or joint-speed equivalence is imposed.',
            'Received pose differences and motor reports corroborate a short reported-state transient but do not establish mechanical or software root cause.',
            'These are adaptive task runs with different actual reference time schedules; per-run distributions are diagnostic, not a strict matched-time causal experiment.',
            'The paced run began in CPV while baseline began in MOVE_J, and initial joint samples differ slightly.']}
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# CPV 节拍修复后的跟踪峰值复核','',
           '比较同一 WSL 中 18:08 原执行器与 19:07 冻结通信修复版本的两次完整抓放。此处只读原始记录，没有修改控制律、检查条件或发送任何实机指令。', '',
           '复算：`env -u LD_LIBRARY_PATH ./envs/oscbf/bin/python -I results/cpv_pacing_review_20261107/analyze_review.py --paced results/pick_place_hardware_20261107_190744_717616`。默认输入保留首次分析时的冻结工作区路径；此参数可指向主工作区已归档的同一结果。', '',
           '## 整体与阶段分布','',
           '| 指标（每控制样本的最大关节误差） | 18:08 基线 | 19:07 修复版本 |',
           '|---|---:|---:|']
    for key,label in [('median','中位数 °'),('p95','P95 °'),('p99','P99 °'),('rms','样本 RMS °'),('max','最大值 °')]:
        lines.append(f"| {label} | {baseline['overall_error_deg'][key]:.6f} | {paced['overall_error_deg'][key]:.6f} |")
    lines += ['', '| 阶段 | 基线最大误差 ° | 修复版最大误差 ° | 基线 RMS ° | 修复版 RMS ° | 修复版超 0.01 rad 样本 |',
              '|---|---:|---:|---:|---:|---:|']
    for stage,b in baseline['stages'].items():
        a=paced['stages'][stage]
        lines.append(f"| {stage} | {b['max_joint_error_deg_stats']['max']:.6f} | {a['max_joint_error_deg_stats']['max']:.6f} | {b['max_joint_error_deg_stats']['rms']:.6f} | {a['max_joint_error_deg_stats']['rms']:.6f} | {a['control_rows_above_001rad']} |")
    lines += ['', '最大的上升集中在 approach，而不是每个阶段都翻倍。排除 approach 后，最大关节误差的样本 RMS 仍从 '
              f"{baseline['without_approach_error_deg']['rms']:.6f}° 增至 {paced['without_approach_error_deg']['rms']:.6f}°，最大值从 {baseline['without_approach_error_deg']['max']:.6f}° 增至 {paced['without_approach_error_deg']['max']:.6f}°。因此也不能写成其余阶段完全没有变化。", '',
              '## 峰值与原始报文','',
              f"峰值在 approach 第 {float(paced['peak']['control_row']['time_s']):.6f} 秒、J5，实际记录关节角 {math.degrees(float(paced['peak']['control_row']['q5_rad'])):.6f}°，参考 {math.degrees(float(paced['peak']['control_row']['q5_ref_rad'])):.6f}°。反馈年龄 {float(paced['peak']['control_row']['feedback_age_s'])*1000:.3f} ms。", '',
              f"独立 raw_can.csv 中对应 0x2a7 载荷为 `{paced['peak']['joint5_peak_raw_packet']['data_hex']}`，按固定 SDK 的 int32 毫度格式重解码后与控制记录的 J5 值一致。峰值附近多个连续关节反馈变化，并且 0x254/0x255 电机反馈也出现位置和速度变化；不支持把它仅当作一个过期样本或单个 SDK 缓存值。", '',
              '0x255 的速度字段按固定 SDK 为 int16 × 0.001 rad/s，位置字段按实际解码实现为 int32 × 0.001 rad。保留“电机反馈字段”的称呼，不自行添加未核对的减速比，也不把报文速度直接当作外部测得的关节真实速度。两个反馈族仍来自机器人内部协议，不是两套外部测量。', '',
              f"峰值 ±0.35 s 窗口内，SDK 发送调用最长 {paced['peak']['window_max_sdk_send_duration_ms']:.3f} ms；相邻六轴命令组起始间隔最大 {paced['peak']['window_max_command_group_interval_ms']:.3f} ms。全程最长 SDK 发送调用发生在 {paced['largest_sdk_send_call']['phase']}，长 {paced['largest_sdk_send_call']['duration_ms']:.3f} ms，早于该跟踪峰值 {paced['largest_sdk_send_call']['seconds_before_peak']:.3f} 秒。不能把两个不同时刻的事件直接合并为一个原因。", '',
              f"approach 中有 {paced['stages']['approach']['zero_reference_rate_rows']} 个参考速率为零的控制样本；这是原跟踪误差推进规则暂停参考，不是驱动使能反馈等待。driver_feedback_pauses 为空。超 0.01 rad 的样本累计约 {paced['stages']['approach']['estimated_time_above_001rad_s']:.6f} s（按相邻控制样本时间估计）。", '',
              '## 结论边界','',
              '本次数据支持“局部明显峰值，伴随部分阶段的小幅统计变化”，不能写成跟踪完全未恶化。CAN 段末错误消失与该跟踪峰值必须分别报告。当前证据不足以认定峰值由发送节拍修复、USB、机械接触、固件或编码器中的任何一个因素单独造成。', '',
              '两次运行都采用反馈自适应参考推进，不是逐时刻相同日程；本报告的分布用于诊断范围判断。基线初始模式是 MOVE_J，修复版初始模式是 CPV，初始反馈也有小差异，已保留在 JSON。', '',
              '本目录保存峰值窗口的控制、反馈、命令、原始报文及重解码 CSV；`review.json` 含完整分阶段统计、报文定位、来源哈希和时间关系。', '']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))
    print(json.dumps({'baseline_overall':baseline['overall_error_deg'],'paced_overall':paced['overall_error_deg'],
        'paced_peak':paced['peak'],'paced_largest_send':paced['largest_sdk_send_call']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
