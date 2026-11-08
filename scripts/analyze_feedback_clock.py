"""Analyze recorded clock domains around a stopped run; no device operations."""
import argparse
import csv
import json
import math
from pathlib import Path

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    args=parser.parse_args();p=args.result
    summary=json.loads((p/'summary.json').read_text())
    feedback=rows(p/'feedback.csv');raw=rows(p/'raw_can.csv');tx=rows(p/'sdk_tx.csv')
    states=[json.loads(line) for line in (p/'states.jsonl').read_text().splitlines()]
    mono=summary['exception_monotonic_s'];wall=summary['exception_time_s']
    previous=next(r for r in reversed(feedback) if float(r['monotonic_s'])<=mono)
    pairs=[{'from_sequence':int(a['sequence']),'to_sequence':int(b['sequence']),
            'monotonic_gap_s':float(b['monotonic_s'])-float(a['monotonic_s']),
            'kernel_realtime_gap_s':float(b['can_timestamp_s'])-float(a['can_timestamp_s'])}
           for a,b in zip(feedback,feedback[1:])]
    for pair in pairs:pair['clock_gap_difference_s']=pair['kernel_realtime_gap_s']-pair['monotonic_gap_s']
    jump=max(pairs,key=lambda r:abs(r['clock_gap_difference_s']))
    offsets=[{'monotonic_s':s['monotonic_s'],'realtime_minus_monotonic_s':s['time_s']-s['monotonic_s']} for s in states]
    steps=[{'before':a,'after':b,'offset_change_s':b['realtime_minus_monotonic_s']-a['realtime_minus_monotonic_s']}
           for a,b in zip(offsets,offsets[1:])]
    offset_step=max(steps,key=lambda x:abs(x['offset_change_s']))
    result={'status':summary['status'],'error':summary['error'],'completed_stages':summary['completed_stages'],
        'old_kernel_realtime_age_at_exception_s':wall-float(previous['can_timestamp_s']),
        'callback_monotonic_age_at_exception_s':mono-float(previous['monotonic_s']),
        'existing_timeout_s':.2,'largest_adjacent_clock_offset_step':offset_step,
        'largest_kernel_timestamp_step':jump,
        'maximum_complete_feedback_callback_gap_s':max(r['monotonic_gap_s'] for r in pairs),
        'can_error_frames':sum(r['is_error']=='1' for r in raw),
        'sdk_send_errors':sum(bool(r['exception'] or r['comm_error_after_send']) for r in tx),
        'stop_can_requests':[r for r in tx if r['arbitration_id']=='0x150' and float(r['send_start_s'])>=mono],
        'post_stop_controller_status':summary['post_stop_state']['arm']['data'],
        'post_stop_q_deg':[math.degrees(v) for v in summary['post_stop_state']['joint']['data']],
        'conclusion':'Feedback continued in monotonic time. A CLOCK_REALTIME step made the epoch-age expression exceed the existing timeout and triggered the original emergency-stop path.',
        'boundary':'Kernel receive and callback times are not measurements of sensor-to-actuator end-to-end latency; the service that changed CLOCK_REALTIME is not established by this log.'}
    (p/'feedback_clock_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    text=['# 本次准备动作停止：时钟域分析','',
          f"运行状态：{summary['status']}。仅完成 {summary['completed_stages']}，完整 OSCBF 尚未开始。",'',
          '| 量 | 记录值 |','|---|---:|',
          f"| 实时时钟相对单调时钟的跳变 | {offset_step['offset_change_s']:.6f} s |",
          f"| 异常时旧表达式算出的 CAN 帧年龄 | {result['old_kernel_realtime_age_at_exception_s']*1000:.3f} ms |",
          f"| 同一最新完整反馈的回调年龄 | {result['callback_monotonic_age_at_exception_s']*1000:.3f} ms |",
          f"| 全程最大完整反馈回调间隔 | {result['maximum_complete_feedback_callback_gap_s']*1000:.3f} ms |",
          f"| CAN 错误帧 | {result['can_error_frames']} |",'',
          '原有0.2秒超时条件受到CLOCK_REALTIME跳变影响。单调时钟记录显示反馈连续，异常时最新完整反馈只有约3ms，因此本次不是记录所显示的真实200ms断流。随后程序确实发送了电子急停。', '',
          '代码修订把原有新鲜度判断统一到接收回调的单调时钟，时限不变。CAN原始实时时间戳继续归档，新增kernel_realtime_age_s用于诊断；不再把它用于控制超时判定。控制器停止状态、驱动故障、原急停处理均保留。', '',
          '本次还没有复位或重新运行。实机恢复需要现场停稳/支撑确认与实际姿态核对，不能从旧计划自动续跑。']
    (p/'CLOCK_ANALYSIS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
