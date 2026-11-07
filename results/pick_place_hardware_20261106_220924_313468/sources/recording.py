"""Passive task diagnostics: independent CAN capture and timestamped SDK states.

This module never sends CAN frames or changes the controller's decisions.
"""
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import can


def interface_stats():
    base = Path('/sys/class/net/can0/statistics')
    return {key: int((base/key).read_text()) for key in
            ('rx_packets', 'rx_bytes', 'rx_errors', 'rx_dropped',
             'tx_packets', 'tx_bytes', 'tx_errors', 'tx_dropped')}


def snapshot(device):
    """Read caches, including faults/stale timestamps, without state() checks."""
    def message(value):
        if value is None:
            return None
        msg = value.msg
        return {'timestamp_s': value.timestamp,
                'data': msg.to_dict() if hasattr(msg, 'to_dict') else list(msg)}

    robot = device.robot
    return {'time_s': time.time(), 'monotonic_s': time.monotonic(),
            'phase': device.phase, 'joint': message(robot.get_joint_angles()),
            'arm': message(robot.get_arm_status()),
            'drivers': [message(robot.get_driver_states(j)) for j in range(1, 7)],
            'motors': [message(robot.get_motor_states(j)) for j in range(1, 7)],
            'communication_error': str(robot.get_comm_error()) if robot.has_comm_error() else None}


class Tee:
    def __init__(self, original, file):
        self.original, self.file = original, file

    def write(self, value):
        self.original.write(value)
        return self.file.write(value)

    def flush(self):
        self.original.flush()
        self.file.flush()


class TaskRecording:
    def __init__(self, out, root, extra_sources=()):
        self.out, self.device = out, None
        self.stop = threading.Event()
        self.error = None
        self.frames, self.states = 0, 0
        self.tx_rows, self.comm, self.original_send = [], None, None
        self.original_stdout, self.original_stderr = sys.stdout, sys.stderr
        self.console = (out/'console.log').open('w', buffering=1)
        self.events = (out/'events.jsonl').open('w', buffering=1)
        sys.stdout = Tee(sys.stdout, self.console)
        sys.stderr = Tee(sys.stderr, self.console)
        self.meta = {'started_at': datetime.now().astimezone().isoformat(),
                     'argv': sys.argv, 'python': sys.executable,
                     'versions': {name: version(name) for name in ('pyAgxArm', 'python-can')},
                     'python_version': sys.version,
                     'interface_before': interface_stats(),
                     'state_period_s': .05,
                     'raw_direction_note': 'SDK disables local loopback: raw_can.csv is received traffic; sdk_tx.csv records send calls, not proof of firmware execution',
                     'state_note': 'SDK cache snapshots have individual message timestamps; raw_can.csv preserves every captured frame'}
        (out/'sources').mkdir()
        for rel in ('scripts/run_pick_place.py', 'piper_oscbf/device.py', 'piper_oscbf/recording.py', *extra_sources):
            shutil.copy2(root/rel, out/'sources'/Path(rel).name)
        (out/'can_interface_before.json').write_text(subprocess.check_output(
            ['ip','-j','-details','-statistics','link','show','can0'],text=True))
        self.bus = can.Bus(interface='socketcan', channel='can0', receive_own_messages=False,
                           ignore_config=True)
        self.worker = threading.Thread(target=self._record, name='passive-can-recording', daemon=True)
        self.worker.start()
        self.mark('recording_started', metadata=self.meta)
        (out/'recording.json').write_text(json.dumps(self.meta, ensure_ascii=False, indent=2)+'\n')

    def bind_connected_device(self, device):
        self.device = device
        self.comm = device.robot.get_context().get_comm()
        self.original_send = self.comm.send

        def recorded_send(msg, *args, **kwargs):
            row = {'time_s': time.time(), 'send_start_s': time.monotonic(),
                   'phase': device.phase, 'arbitration_id': hex(msg.arbitration_id),
                   'dlc': msg.dlc, 'data_hex': msg.data.hex(), 'exception': None}
            try:
                return self.original_send(msg, *args, **kwargs)
            except Exception as exc:
                row['exception'] = repr(exc)
                raise
            finally:
                row['send_end_s'] = time.monotonic()
                row['comm_error_after_send'] = str(self.comm.last_error) if self.comm.last_error else None
                self.tx_rows.append(row)

        self.comm.send = recorded_send
        self.mark('sdk_tx_observation_attached')

    def mark(self, event, **values):
        row = {'event': event, 'time_s': time.time(), 'monotonic_s': time.monotonic(), **values}
        self.events.write(json.dumps(row, ensure_ascii=False)+'\n')

    def _record(self):
        try:
            with (self.out/'raw_can.csv').open('w', newline='') as raw, \
                    (self.out/'states.jsonl').open('w') as states:
                writer = csv.writer(raw)
                writer.writerow(('can_timestamp_s', 'receive_monotonic_s', 'phase', 'arbitration_id',
                                 'is_rx', 'is_error', 'is_remote', 'is_extended', 'dlc', 'data_hex'))
                next_state = next_flush = time.monotonic()
                while not self.stop.is_set():
                    frame = self.bus.recv(.02)
                    now = time.monotonic()
                    phase = self.device.phase if self.device else 'before_connect'
                    if frame is not None:
                        writer.writerow((frame.timestamp, now, phase, hex(frame.arbitration_id),
                                         int(frame.is_rx), int(frame.is_error_frame), int(frame.is_remote_frame),
                                         int(frame.is_extended_id), frame.dlc, frame.data.hex()))
                        self.frames += 1
                    if self.device and now >= next_state:
                        states.write(json.dumps(snapshot(self.device), ensure_ascii=False)+'\n')
                        self.states += 1
                        next_state = now+.05
                    if now >= next_flush:
                        raw.flush()
                        states.flush()
                        next_flush = now+1.
        except Exception:
            self.error = traceback.format_exc()
            print('诊断记录异常（控制器未改变）：', self.error, flush=True)

    def finish(self):
        self.stop.set()
        self.worker.join(timeout=1.)
        self.bus.shutdown()
        if self.original_send:
            self.comm.send = self.original_send
        with (self.out/'sdk_tx.csv').open('w',newline='') as stream:
            fields = ('time_s','send_start_s','phase','arbitration_id','dlc','data_hex',
                      'exception','send_end_s','comm_error_after_send')
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.tx_rows)
        (self.out/'can_interface_after.json').write_text(subprocess.check_output(
            ['ip','-j','-details','-statistics','link','show','can0'],text=True))
        self.meta.update(ended_at=datetime.now().astimezone().isoformat(),
                         raw_frames=self.frames, state_snapshots=self.states,
                         sdk_send_calls=len(self.tx_rows),
                         recorder_error=self.error, worker_stopped=not self.worker.is_alive(),
                         interface_after=interface_stats())
        self.meta['interface_delta'] = {k: self.meta['interface_after'][k]-v
                                        for k, v in self.meta['interface_before'].items()}
        self.mark('recording_finished', metadata=self.meta)
        (self.out/'recording.json').write_text(json.dumps(self.meta, ensure_ascii=False, indent=2)+'\n')
        self.events.close()
        sys.stdout, sys.stderr = self.original_stdout, self.original_stderr
        self.console.close()
