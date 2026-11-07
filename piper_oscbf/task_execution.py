"""Filtered task stage execution using the existing CPV transport and stops.

No hardware is created here. The caller owns the single device and its existing
exception/stop handler; errors, including their diagnostics, propagate unchanged.
"""
import time
import numpy as np
from piper_oscbf.task_filter import TaskFilterError, quantize_sdk
from piper_oscbf.task_reference import StageReference, PhaseGovernor, ReferenceBlockedError


class TaskExecutionError(RuntimeError):
    def __init__(self, reason, diagnostics):
        self.reason,self.diagnostics = reason,diagnostics
        super().__init__(f'{reason}: {diagnostics}')


def _cap(cfg, stage):
    return float(cfg.get('stage_velocity_limits_rad_s', {}).get(stage, cfg['joint_velocity_limit_rad_s']))


def _gripper_message(gripper):
    value = gripper.get_gripper_status()
    if value is None or not 0 <= time.time()-value.timestamp <= .2:
        raise RuntimeError('Gripper feedback unavailable or stale')
    msg = value.msg
    if not np.isfinite(msg.value) or not 0 <= float(msg.value) <= .1:
        raise RuntimeError('Gripper opening is not a finite physical width')
    faults = [key for key,active in msg.foc_status.to_dict().items()
              if active and key not in ('driver_enable_status','homing_status')]
    if faults:
        raise RuntimeError(f'Gripper fault: {faults}')
    return value


def _checked_send(device, result, session, sample):
    """Recheck the exact SDK-rounded request before the shared send entry."""
    requested = np.asarray(result['u_sdk'], dtype=float)
    if requested.shape != (6,) or not np.all(np.isfinite(requested)) or not result['converged']:
        raise TaskFilterError('invalid_filtered_request', result)
    predicted = quantize_sdk(requested)
    margin = np.asarray(result['b_physical']) - np.asarray(result['G'])@predicted
    if not np.array_equal(predicted, requested) or float(margin.min()) < -session.filter.residual_tol:
        result['pre_send_margin'] = margin
        raise TaskFilterError('pre_send_quantized_request_violation', result)
    ages = {'monotonic_age_s':time.monotonic()-sample['monotonic_s'],
            'can_age_s':time.time()-sample['can_timestamp_s']}
    if any(not np.isfinite(v) or v > .2 for v in ages.values()):
        raise TaskExecutionError('used_feedback_stale_before_send',ages)
    device.state()  # Reapply the existing feedback/driver checks after computation.
    sent = np.asarray(device.velocity(predicted.tolist()), dtype=float)
    if sent.shape != (6,) or not np.array_equal(sent, predicted):
        result['actual_sdk_request'] = sent
        raise TaskFilterError('device_changed_validated_request', result)
    # Preserve margins for the request that actually entered PiperDevice.
    result['u_sdk'], result['applied_margin'] = sent, margin
    return sent


def _control_row(device, sample, status, q, q_ref, result, session, stage, started,
                 now, *, phase=None, kind='motion'):
    sent = np.asarray(result['u_sdk'])
    nominal = np.asarray(result['u_nom'])
    correction = sent-nominal
    tcp, reference_tcp = np.asarray(result['tcp_transform']), np.asarray(result['reference_tcp_transform'])
    geom_count = session.filter.geometry_constraint_count
    nominal_margin = np.asarray(result['nominal_margin'])
    q_nominal_rounded = quantize_sdk(nominal)
    row = {'stage':stage, 'kind':kind, 'time_s':now-started, 'monotonic_s':now,
        'reference_time_s':phase.phase_s if phase is not None else 0.,
        'phase_rate':phase.rate if phase is not None else 0., 'method':session.method,
        'tcp_position_error_m':float(np.linalg.norm(reference_tcp[:3,3]-tcp[:3,3])),
        'orientation_error_rad':float(result['orientation_error_rad']),
        'correction_joint_l2_rad_s':float(np.linalg.norm(correction)),
        'filter_correction_joint_l2_rad_s':float(np.linalg.norm(sent-q_nominal_rounded)),
        'correction_task_l2':float(np.linalg.norm(np.asarray(result['jacobian'])@correction)),
        'constraint_active':bool(result['filter_active']),
        'geometry_active':bool(np.any(nominal_margin[:geom_count] < -session.filter.residual_tol)),
        'min_h_m':float(result['min_h_m']),
        'min_physical_distance_m':float(result['min_physical_distance_m']),
        'min_applied_margin':float(np.min(result['applied_margin'])),
        'qp_solved':bool(result['qp_solved']), 'qp_converged':bool(result['converged']),
        'qp_iterations':int(result['iterations']), 'qp_ms':float(result['solve_ms']),
        'solver_status':str(result['solver_status']),
        'sigma_min':float(result['sigma_min']), 'metric_condition':float(result['metric_condition']),
        'feedback_age_s':time.time()-sample['can_timestamp_s'],
        'feedback_monotonic_age_s':now-sample['monotonic_s'],
        'sample_sequence':sample['sequence'], 'can_timestamp_s':sample['can_timestamp_s'],
        'max_joint_error_rad':float(np.max(np.abs(q_ref-q))),
        'progress_rate':phase.rate if phase is not None else 0.,
        'mode_feedback':int(status['mode_feedback']), 'arm_status':int(status['arm_status']),
        'send_group_start_s':device.commands[-1]['send_start_s'],
        'send_group_end_s':device.commands[-1]['send_end_s']}
    for axis,index in zip('xyz',range(3)):
        row[f'tcp_{axis}_m'],row[f'tcp_ref_{axis}_m'] = float(tcp[index,3]),float(reference_tcp[index,3])
    for j in range(1,7):
        row[f'q{j}_rad'],row[f'q{j}_ref_rad'] = float(q[j-1]),float(q_ref[j-1])
        row[f'u{j}_sdk_rad_s'],row[f'u{j}_nom_rad_s'],row[f'u{j}_qp_rad_s'] = (
            float(sent[j-1]),float(nominal[j-1]),float(result['u_qp'][j-1]))
    if phase is not None:
        row.update({f'phase_{key}':value for key,value in phase.as_dict().items()})
    return row


def _validated_zero_hold(device, gripper, cfg, session, stage, duration, task_feedback,
                         *, after=None, require_cpv=True, anchor=None):
    """Normal zero groups remain subject to geometry checks and the same pacer."""
    device.phase = 'zero_velocity'
    started, seq, last_status = time.monotonic(), after, None
    cap = _cap(cfg,stage)
    while time.monotonic()-started < duration:
        device.wait_for_velocity_slot()
        sample,status = task_feedback(device,after=seq)
        seq,last_status = sample['sequence'],status
        if require_cpv and int(status['mode_feedback']) != 5:
            raise RuntimeError('CPV mode lost during normal filtered hold')
        q = np.asarray(device.angles(sample), dtype=float)
        if anchor is not None and np.max(np.abs(q-anchor)) > cfg['arrival_tolerance_rad']:
            raise RuntimeError('Arm moved outside arrival tolerance during fixed-pose gripper hold')
        width = float(_gripper_message(gripper).msg.value)
        result = session.validate_hold(q,width,cap,stage)
        _checked_send(device,result,session,sample)
        now = time.monotonic()
        row = _control_row(device,sample,status,q,q,result,session,stage,started,now,kind='normal_zero_hold')
        session.add_hold(row,result)
    if last_status is not None and int(last_status['mode_feedback']) != 5:
        raise RuntimeError('Controller did not enter CPV during initial normal hold')
    return seq


def filtered_arm_step(device, gripper, step, cfg, records, session, task_feedback):
    stage,cap = step['name'],_cap(cfg,step['name'])
    reference = StageReference(step,cap)
    first,_ = device.state()
    if np.max(np.abs(np.asarray(device.angles(first))-reference.points[0])) > cfg['tracking_tolerance_rad']:
        raise RuntimeError(f'{stage}: actual state differs from planned start; no stage motion sent')
    device.robot.set_motion_mode('cpv')
    # The existing executor allows this .2s hold for mode feedback to settle.
    seq = _validated_zero_hold(device,gripper,cfg,session,stage,.2,task_feedback,
                               after=first['sequence'],require_cpv=False)
    device.phase = stage
    governor = PhaseGovernor(reference,cfg['arrival_tolerance_rad'],**cfg.get('reference_governor',{}))
    started,printed = time.monotonic(),-2.
    states,primary_error = [],None
    try:
        while True:
            device.wait_for_velocity_slot()
            pause_before=getattr(device,'driver_pause_s',0.)
            sample,status = task_feedback(device,after=seq)
            seq = sample['sequence']
            if int(status['mode_feedback']) != 5:
                raise RuntimeError('CPV mode lost')
            now = time.monotonic()
            if getattr(device,'driver_pause_s',0.) > pause_before and governor.previous is not None:
                governor.previous=now
            q = np.asarray(device.angles(sample),dtype=float)
            try:
                phase = governor.update(q,now=now)
            except ReferenceBlockedError as exc:
                states.append(exc.state.as_dict())
                raise
            states.append(phase.as_dict())
            q_ref,qd_ref = reference.sample(phase.phase_s)
            width = float(_gripper_message(gripper).msg.value)
            result = session.filter_step(q,q_ref,qd_ref*phase.rate,width,cap,stage)
            _checked_send(device,result,session,sample)
            row = _control_row(device,sample,status,q,q_ref,result,session,stage,started,now,phase=phase)
            row['reference_signature'] = reference.signature
            records.append(row)
            session.add_control(row,result)  # same row object; scalar/array indices stay aligned
            if now-started-printed >= 2.:
                print(f'{stage} [{session.method}]: phase={phase.phase_s:.2f}/{reference.duration:.2f}s, '
                      f'joint error={np.rad2deg(row["max_joint_error_rad"]):.3f}deg, '
                      f'min h={result["min_h_m"]:.6f}m, QP={result["solver_status"]}',flush=True)
                printed = now-started
            if phase.done:
                break
        _validated_zero_hold(device,gripper,cfg,session,stage,.5,task_feedback,after=seq)
        return phase
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            session.add_schedule(stage,reference,states)
        except Exception as record_error:
            if primary_error is None:
                raise
            # A logging problem must not replace a filter/blocked exception and
            # erase the diagnostics used by the caller's existing stop handler.
            primary_error.schedule_record_error = str(record_error)


def filtered_gripper_step(device, gripper, step, cfg, rows, session, task_feedback):
    stage,cap = step['name'],_cap(cfg,step['name'])
    device.phase = stage
    initial,_ = device.state()
    anchor = np.asarray(device.angles(initial),dtype=float)
    initial_width = float(_gripper_message(gripper).msg.value)
    session.check_gripper(anchor,initial_width,float(step['width_m']),stage)
    settle_tolerance = float(cfg.get('gripper_settle_tolerance_m',.0005))
    if not np.isfinite(settle_tolerance) or settle_tolerance <= 0:
        raise ValueError('Positive finite engineering gripper settling tolerance required')
    start_index = len(rows)
    started,seq = time.monotonic(),initial['sequence']
    last_value = None

    def record_gripper(value, command_sent):
        msg = value.msg
        rows.append({'stage':stage,'can_timestamp_s':value.timestamp,
            'command_width_m':step['width_m'],'width_m':msg.value,
            'command_force_n':step['force_n'],'force_n':msg.force,
            'command_sent':command_sent,'settle_tolerance_m':settle_tolerance,
            'settle_window_range_m':None,'settle_distinct_samples':None,'settled':None})

    while time.monotonic()-started < 1.5:
        device.wait_for_velocity_slot()
        sample,status = task_feedback(device,after=seq)
        seq = sample['sequence']
        q = np.asarray(device.angles(sample),dtype=float)
        if np.max(np.abs(q-anchor)) > cfg['arrival_tolerance_rad']:
            raise RuntimeError('Arm moved outside arrival tolerance during fixed-pose gripper actuation')
        width = float(_gripper_message(gripper).msg.value)
        result = session.validate_hold(q,width,cap,stage)
        if int(status['mode_feedback']) == 5:
            _checked_send(device,result,session,sample)
            now = time.monotonic()
            row = _control_row(device,sample,status,q,q,result,session,stage,started,now,kind='gripper_zero_hold')
            session.add_hold(row,result)
        gripper.move_gripper_m(value=step['width_m'],force=step['force_n'])
        time.sleep(.05)
        last_value = _gripper_message(gripper)
        record_gripper(last_value,True)
    final,status = task_feedback(device,after=seq)
    q = np.asarray(device.angles(final),dtype=float)
    if np.max(np.abs(q-anchor)) > cfg['arrival_tolerance_rad']:
        raise RuntimeError('Arm moved outside arrival tolerance at gripper completion')
    last_value = _gripper_message(gripper)
    width = float(last_value.msg.value)
    record_gripper(last_value,False)
    # Use only the observations collected above: no extra motion, waits or
    # retries, and no requirement to reach the closing command through an object.
    window,seen = [],set()
    for row in reversed(rows[start_index:]):
        stamp = float(row['can_timestamp_s'])
        if stamp not in seen:
            window.append({'timestamp_s':stamp,'width_m':float(row['width_m'])})
            seen.add(stamp)
        if len(window)==3:
            break
    window.reverse()
    width_range = max(x['width_m'] for x in window)-min(x['width_m'] for x in window)
    stable = (len(window)==3 and all(a['timestamp_s']<b['timestamp_s'] for a,b in zip(window,window[1:]))
              and width_range <= settle_tolerance)
    rows[-1].update(settle_window_range_m=width_range,settle_distinct_samples=len(window),settled=stable)
    if not stable:
        raise TaskExecutionError('gripper_not_settled',{'stage':stage,'window':window,
            'width_range_m':width_range,'tolerance_m':settle_tolerance,
            'rule':'last three distinct timestamps, engineering stability guard; not a grasp-success classifier'})
    session.validate_hold(q,width,cap,stage)
    session.after_gripper(stage,q,width)
    print(f'{stage}: actual gripper width={width*1000:.1f}mm force={last_value.msg.force:.2f}N',flush=True)
