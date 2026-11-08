"""Filtered task stage execution using the existing CPV transport and stops.

No hardware is created here. The caller owns the single device and its existing
exception/stop handler; errors, including their diagnostics, propagate unchanged.
"""
import time
import numpy as np
from piper_oscbf.task_filter import quantize_sdk
from piper_oscbf.task_reference import StageReference, PhaseGovernor, ReferenceBlockedError


def _cap(cfg, stage):
    return float(cfg.get('stage_velocity_limits_rad_s', {}).get(stage, cfg['joint_velocity_limit_rad_s']))


def _gripper_message(device, gripper):
    value=gripper.get_gripper_status()
    if value is None or device.message_age(0x2A8)>.2:
        raise RuntimeError('Gripper feedback unavailable')
    faults=[key for key,active in value.msg.foc_status.to_dict().items()
            if active and key not in ('driver_enable_status','homing_status')]
    if faults:raise RuntimeError(f'Gripper fault: {faults}')
    return value


def _send(device, result):
    sent=np.asarray(device.velocity(np.asarray(result['u_sdk']).tolist()),dtype=float)
    result['u_sdk']=sent
    result['applied_margin']=np.asarray(result['b_physical'])-np.asarray(result['G'])@sent
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
        'qp_solved':bool(result['qp_solved']), 'qp_converged':result['converged'],
        'qp_iterations':int(result['iterations']), 'qp_ms':float(result['solve_ms']),
        'solver_status':str(result['solver_status']),
        'sigma_min':float(result['sigma_min']), 'metric_condition':float(result['metric_condition']),
        'feedback_age_s':time.monotonic()-sample['monotonic_s'],
        'kernel_realtime_age_s':time.time()-sample['can_timestamp_s'],
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


def _zero_hold(device, gripper, cfg, session, stage, duration, task_feedback, *, after=None):
    device.phase='zero_velocity'
    started,seq=time.monotonic(),after
    cap=_cap(cfg,stage)
    while time.monotonic()-started<duration:
        device.wait_for_velocity_slot()
        sample,status=task_feedback(device,after=seq)
        seq=sample['sequence']
        q=np.asarray(device.angles(sample),dtype=float)
        width=float(_gripper_message(device,gripper).msg.value)
        result=session.observe_hold(q,width,cap,stage)
        _send(device,result)
        row=_control_row(device,sample,status,q,q,result,session,stage,started,
                         time.monotonic(),kind='zero_velocity_observation')
        session.add_hold(row,result)
    return seq


def filtered_arm_step(device, gripper, step, cfg, records, session, task_feedback):
    stage,cap = step['name'],_cap(cfg,step['name'])
    reference = StageReference(step,cap)
    first,_ = device.state()
    if np.max(np.abs(np.asarray(device.angles(first))-reference.points[0])) > cfg['tracking_tolerance_rad']:
        raise RuntimeError(f'{stage}: actual state differs from planned start; no stage motion sent')
    device.robot.set_motion_mode('cpv')
    # The existing executor allows this .2s hold for mode feedback to settle.
    seq = _zero_hold(device,gripper,cfg,session,stage,.2,task_feedback,
                               after=first['sequence'])
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
            width = float(_gripper_message(device,gripper).msg.value)
            result = session.filter_step(q,q_ref,qd_ref*phase.rate,width,cap,stage)
            _send(device,result)
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
        _zero_hold(device,gripper,cfg,session,stage,.5,task_feedback,after=seq)
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
    # Same 1.5s/50ms gripper behavior as the original baseline. A zero CPV
    # request is not position holding; actual geometry is logged, not gated.
    stage,cap=step['name'],_cap(cfg,step['name'])
    device.phase=stage
    started=time.monotonic()
    while time.monotonic()-started<1.5:
        sample,status=device.state()
        q=np.asarray(device.angles(sample),dtype=float)
        if int(status['mode_feedback'])==5:
            width=float(_gripper_message(device,gripper).msg.value)
            result=session.observe_hold(q,width,cap,stage)
            _send(device,result)
            row=_control_row(device,sample,status,q,q,result,session,stage,started,
                             time.monotonic(),kind='gripper_zero_velocity_observation')
            session.add_hold(row,result)
        gripper.move_gripper_m(value=step['width_m'],force=step['force_n'])
        time.sleep(.05)
        value=_gripper_message(device,gripper);msg=value.msg
        rows.append({'stage':stage,'can_timestamp_s':value.timestamp,
            'command_width_m':step['width_m'],'width_m':msg.value,
            'command_force_n':step['force_n'],'force_n':msg.force})
    final,_=device.state()
    q=np.asarray(device.angles(final),dtype=float)
    session.after_gripper(stage,q,float(msg.value))
    print(f'{stage}: actual gripper width={msg.value*1000:.1f}mm force={msg.force:.2f}N',flush=True)
