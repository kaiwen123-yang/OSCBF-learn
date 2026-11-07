"""Read recorded runs and write only the new WSL run's comparison artifacts."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path('/home/kaiwen/OSCBF-learn')
NAMES = ['pick_place_hardware_20261106_220347_670955',
         'pick_place_hardware_20261106_220924_313468',
         'pick_place_hardware_20261107_180811_947624']
OUT = ROOT / 'results' / NAMES[-1]

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'samples': len(a), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99)),
            'max': float(a.max()), 'over_20ms': int((a > 20).sum()),
            'over_100ms': int((a > 100).sum())}

result = []
for name in NAMES:
    p = ROOT / 'results' / name
    a = json.loads((p / 'analysis.json').read_text())
    s = json.loads((p / 'summary.json').read_text())
    r = json.loads((p / 'recording.json').read_text())
    can_before = (json.loads((p / 'can_interface_before.json').read_text())[0]['linkinfo']
                  if (p / 'can_interface_before.json').exists() else None)
    can_after = (json.loads((p / 'can_interface_after.json').read_text())[0]['linkinfo']
                 if (p / 'can_interface_after.json').exists() else None)
    feedback, control, commands = [rows(p / f'{key}.csv') for key in ('feedback', 'control', 'commands')]
    gaps = [1000 * (float(b['can_timestamp_s']) - float(v['can_timestamp_s']))
            for v, b in zip(feedback, feedback[1:])]
    callback_gaps = [1000 * (float(b['monotonic_s']) - float(v['monotonic_s']))
                     for v, b in zip(feedback, feedback[1:])]
    cycles = [1000 * (float(b['monotonic_s']) - float(v['monotonic_s']))
              for v, b in zip(control, control[1:]) if v['stage'] == b['stage']]
    send_durations = [1000 * (float(v['send_end_s']) - float(v['send_start_s'])) for v in commands]
    errors = rows(p / 'can_errors.csv')
    classes = {1: 'tx_timeout', 2: 'lost_arbitration', 4: 'controller', 8: 'protocol',
               16: 'transceiver', 32: 'ack', 64: 'bus_off', 128: 'bus_error',
               256: 'restarted', 512: 'error_counter'}
    err_counts = {label: sum(bool(int(v['arbitration_id'], 16) & flag) for v in errors)
                  for flag, label in classes.items()}
    initial = next(json.loads(line) for line in (p / 'events.jsonl').read_text().splitlines()
                   if json.loads(line)['event'] == 'connected')['state']
    result.append({
        'name': name, 'platform': 'WSL2' if name == NAMES[-1] else 'native Ubuntu historical',
        'status': s['status'], 'task_duration_s': a['task_duration_s'],
        'completed_stages': s['completed_stages'],
        'speed_scale': s['speed_scale'], 'descent_speed_scale': s['descent_speed_scale'],
        'initial_mode': initial['arm']['data']['mode_feedback'],
        'initial_joint_deg': np.rad2deg(initial['joint']['data']).tolist(),
        'interface_delta': r['interface_delta'], 'raw_frames': a['raw_frames'],
        'can_state_before': can_before['info_data']['state'] if can_before else None,
        'can_state_after': can_after['info_data']['state'] if can_after else None,
        'can_xstats_delta': ({k: v - can_before['info_xstats'].get(k, 0) for k, v in can_after['info_xstats'].items()}
                             if can_before and can_after else None),
        'can_error_frames': a['can_error_frames'], 'can_error_bursts': a['can_error_bursts'],
        'error_flag_frame_counts_nonexclusive': err_counts,
        'feedback_samples': len(feedback),
        'feedback_hz': (len(feedback) - 1) / (float(feedback[-1]['can_timestamp_s']) - float(feedback[0]['can_timestamp_s'])),
        'feedback_gap_ms': distribution(gaps), 'feedback_callback_gap_ms': distribution(callback_gaps),
        'control_cycle_ms': distribution(cycles), 'six_axis_send_group_duration_ms': distribution(send_durations),
        'driver_max_gap_ms': max(a['driver_max_gap_ms'].values()),
        'driver_codes': a['driver_codes'], 'driver_feedback_pauses': a['driver_feedback_pauses'],
        'max_tracking_error_deg': a['max_tracking_error_deg'],
        'sdk_send_calls': a.get('sdk_send_calls'),
        'sdk_send_error_count': len(a['sdk_send_errors']) if 'sdk_send_errors' in a else None,
        'recorder_error': a['recorder_error'],
    })

summary_path = OUT / 'summary.json'
summary = json.loads(summary_path.read_text())
summary['object_transfer_success'] = 'operator_confirmed'
summary['operator_confirmation'] = {
    'time': datetime.now().astimezone().isoformat(),
    'text': '完整抓放成功，动作正常',
    'context': 'After this WSL run: actual pickup, left placement and return confirmed by the onsite operator.'}
summary['post_run_analysis'] = {'analysis_file': 'analysis.json', 'communication_comparison': 'wsl_communication_comparison.json',
                                'report': 'WSL_COMMUNICATION.md'}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
comparison = {
    'created_at': datetime.now().astimezone().isoformat(),
    'scope': 'One authorized WSL hardware pick/place replay; same saved path and both speed scales 1.5. No control-law change; no OSCBF filter in this task.',
    'runs': result,
    'limitations': ['Historical comparison, not repeated randomized platform benchmarking.',
                    'CAN error frames and Linux interface rx_errors/tx_errors are different counters.',
                    'Zero interface drops do not prove end-to-end absence of loss.',
                    'SDK transmit-call success is not per-command firmware execution acknowledgement.',
                    'Segment-end timing coincidence does not establish root cause.'],
}
(OUT / 'wsl_communication_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')

w = result[-1]
lines = ['# WSL Piper 实机复跑通信记录', '',
         f'运行：`{NAMES[-1]}`。用户已确认“完整抓放成功，动作正常”。', '',
         '使用原 MoveIt 路径加关节 CPV 跟踪，全阶段速度倍率 1.5，关节速度上限 0.30 rad/s。11 个阶段全部完成，返回零位附近，保持 CPV 零速度及使能；没有自动复位或追加运动。本任务不含 OSCBF。', '',
         '| 指标 | 原生 Ubuntu 11/6 第一次 | 原生 Ubuntu 11/6 第二次 | WSL 本次 |',
         '|---|---:|---:|---:|']
metrics = [('任务时间（s）', lambda d: f"{d['task_duration_s']:.3f}"),
           ('关节反馈频率（Hz）', lambda d: f"{d['feedback_hz']:.2f}"),
           ('反馈最大间隔（ms）', lambda d: f"{d['feedback_gap_ms']['max']:.3f}"),
           ('SDK 反馈回调最大间隔（ms）', lambda d: f"{d['feedback_callback_gap_ms']['max']:.3f}"),
           ('控制周期中位数（ms）', lambda d: f"{d['control_cycle_ms']['median']:.3f}"),
           ('控制周期 P95（ms）', lambda d: f"{d['control_cycle_ms']['p95']:.3f}"),
           ('控制周期最大值（ms）', lambda d: f"{d['control_cycle_ms']['max']:.3f}"),
           ('控制周期 >20 ms 次数', lambda d: str(d['control_cycle_ms']['over_20ms'])),
           ('六轴命令组发送耗时最大值（ms）', lambda d: f"{d['six_axis_send_group_duration_ms']['max']:.3f}"),
           ('CAN 错误帧', lambda d: str(d['can_error_frames'])),
           ('CAN 错误突发组数', lambda d: str(len(d['can_error_bursts']))),
           ('最大关节跟踪误差（°）', lambda d: f"{d['max_tracking_error_deg']:.3f}"),
           ('SDK 发送异常', lambda d: '未完整记录' if d['sdk_send_error_count'] is None else str(d['sdk_send_error_count']))]
lines += [f"| {label} | " + ' | '.join(fn(d) for d in result) + ' |' for label, fn in metrics]
lines += ['', '## 本次通信结果', '',
          f"采集 {w['raw_frames']} 帧原始 CAN、{w['feedback_samples']} 组六轴反馈、{w['sdk_send_calls']} 次 SDK 发送。记录器无异常，SDK 发送异常为 0，没有驱动反馈等待事件。六轴驱动状态字均为 0x40，未观察到驱动故障或 bus-off。", '',
          'Linux 接口的收发错误、丢弃计数增量均为 0，但原始 CAN 记录中有 35 条错误帧。因此不能把接口计数为零写成总线零错误。', '',
          'CAN 专项计数中 error_warning 增加 9、error_passive 增加 6，bus_off 为 0；结束时恢复 ERROR-ACTIVE。CAN 总线错误状态与六轴电机驱动状态是不同指标。', '',
          '本次有一次超过 20 ms 的控制周期，最大 24.997 ms；六轴命令组发送耗时最大 24.808 ms。SDK 反馈回调间隔最大 27.609 ms，而按 CAN 接收时间戳计算的六轴反馈最大间隔为 9.480 ms。这几个量分别描述应用处理与接收时间戳间隔，不能直接当作绝对 USB 单程延迟；本轮没有保证严格恒定 100 Hz。', '',
          '两组错误均发生在段末 zero_velocity 阶段：', '',
          '| 前一运动段 | 错误帧 | 持续时间（ms） | 前组发送结束到第一组零速（ms） | 最大报告 TX 错误计数 |',
          '|---|---:|---:|---:|---:|']
for b in w['can_error_bursts']:
    lines.append(f"| {b['after_stage']} | {b['frames']} | {b['duration_s']*1000:.3f} | {b['last_command_end_to_first_zero_start_ms']:.3f} | {b['max_reported_tx_error_counter']} |")
lines += ['', '本轮完整任务与双向通信在 WSL 下可运行，现场确认抓放成功。此前原生 Ubuntu 也有段末 CAN 突发，当前仍存在同类现象。本次错误帧较少不足以证明 WSL 更稳定，也不能仅凭这三次运行确定根因。', '',
          '后续应单独验证段末零速是否需要纳入原有 10 ms 发送节拍；本轮没有修改该逻辑。Windows 不同负载下的长期运行、任意 USB 热重连、完整任务 OSCBF 均不在本轮证据范围内。', '',
          '## 原始记录', '',
          '- `summary.json`：结果、现场确认、末态。',
          '- `raw_can.csv` / `can_errors.csv`：原始接收/错误帧及错误子集。',
          '- `sdk_tx.csv` / `commands.csv`：发送调用、六轴命令组起止时间。',
          '- `feedback.csv` / `control.csv` / `driver_feedback.csv`：真实反馈、控制周期和驱动状态。',
          '- `recording.json` / `events.jsonl` / `states.jsonl`：接口计数、阶段事件和状态快照。',
          '- `analysis.json` / `wsl_communication_comparison.json`：派生统计及历史比较。',
          '- `sources/`：本次控制与分析代码快照。', '']
(OUT / 'WSL_COMMUNICATION.md').write_text('\n'.join(lines))
print(json.dumps([{k:d[k] for k in ['name','feedback_hz','feedback_gap_ms','feedback_callback_gap_ms','control_cycle_ms','six_axis_send_group_duration_ms','can_error_frames','error_flag_frame_counts_nonexclusive']} for d in result], ensure_ascii=False, indent=2))
