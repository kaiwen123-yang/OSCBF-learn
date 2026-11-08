"""Execute saved MoveIt joint paths through the already verified Piper SDK CPV.

No ROS hardware driver is started. Method selects the original joint-tracking
baseline or CBF/OSCBF filtering on arm motion. Default prints the saved plan.
"""
import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_piper_plane_hardware import save_csv

JOINTS = [f"joint{i}" for i in range(1,7)]


def path_arrays(step, cap):
    trajectory = step["trajectory"]["joint_trajectory"]
    order = [trajectory["joint_names"].index(j) for j in JOINTS]
    points = trajectory["points"]
    q = np.array([[p["positions"][i] for i in order] for p in points])
    times = np.array([p["time_from_start"]["sec"]+p["time_from_start"]["nanosec"]*1e-9 for p in points])
    # Reserve half the tested velocity cap for feedback correction. Interpolate
    # every planned point; never replace the collision-checked path by endpoints.
    durations = np.maximum(np.diff(times), np.max(np.abs(np.diff(q,axis=0)),axis=1)/(cap*.5))
    durations = np.maximum(durations,.01)
    return np.r_[0.,np.cumsum(durations)], q


def task_feedback(device, after):
    from piper_oscbf.device import MotorEnableUnconfirmed
    try:
        return device.state(after=after)
    except MotorEnableUnconfirmed:
        # One raw J6 frame reported 4414.7 V/28672 C during the last task.
        # Stop velocity while waiting for fresh confirmation within the existing
        # 200 ms feedback window. Never re-enable a motor or ignore a fault bit.
        start=time.monotonic()
        event={"phase":device.phase,"initial_check":device.last_driver_check,"recovered":False}
        device.driver_feedback_pauses.append(event)
        print("驱动使能反馈待确认：参考暂停，发送零速度等待新反馈。",flush=True)
        first_zero = True
        while True:
            device.velocity([0.]*6, immediate=first_zero)
            first_zero = False
            time.sleep(.01)
            try:
                result=device.state()
                break
            except MotorEnableUnconfirmed:
                if time.monotonic()-start>=.2:
                    raise
        event.update(recovered=True,duration_s=time.monotonic()-start)
        device.driver_pause_s+=event["duration_s"]
        return result


def arm_step(device, step, cfg, records):
    cap = cfg.get("stage_velocity_limits_rad_s",{}).get(step["name"],cfg["joint_velocity_limit_rad_s"])
    times, points = path_arrays(step,cap)
    if len(points)==1:
        # A one-point prepared-stage plan is a hold with real arrival checking.
        points=np.repeat(points,2,axis=0);times=np.array([0.,.01])
    first,_ = device.state()
    if np.max(np.abs(np.asarray(device.angles(first))-points[0])) > cfg["tracking_tolerance_rad"]:
        raise RuntimeError(f"{step['name']}: actual state differs from planned start; replan from actual state")
    device.robot.set_motion_mode("cpv")
    device.zero(.2)
    device.phase = step["name"]
    phase, rate = 0., 1.
    start = previous = time.monotonic()
    seq, printed = first["sequence"], -2.
    while True:
        # Wait before reading feedback so the control calculation uses a fresh
        # sample. velocity() also enforces this same cross-stage schedule.
        device.wait_for_velocity_slot()
        pause_before=device.driver_pause_s
        sample,status = task_feedback(device,after=seq)
        if device.driver_pause_s>pause_before:
            previous=time.monotonic()
        if int(status["mode_feedback"]) != 5:
            raise RuntimeError("CPV mode lost")
        seq = sample["sequence"]
        now = time.monotonic()
        phase = min(times[-1],phase+(now-previous)*rate)
        previous = now
        i = min(len(times)-2,max(0,int(np.searchsorted(times,phase,side="right")-1)))
        frac = (phase-times[i])/(times[i+1]-times[i])
        desired = points[i]*(1-frac)+points[i+1]*frac
        velocity = (points[i+1]-points[i])/(times[i+1]-times[i])*rate if phase<times[-1] else np.zeros(6)
        q = np.asarray(device.angles(sample))
        error = desired-q
        requested = velocity+2.*error
        requested /= max(1.,float(np.max(np.abs(requested)))/cap)
        sent = device.velocity(requested)
        row={"stage":step["name"],"time_s":now-start,"monotonic_s":now,"reference_time_s":phase,
             "sample_sequence":seq,"can_timestamp_s":sample["can_timestamp_s"],
             "feedback_age_s":time.monotonic()-sample["monotonic_s"],
             "kernel_realtime_age_s":time.time()-sample["can_timestamp_s"],
             "feedback_monotonic_age_s":now-sample["monotonic_s"],
             "send_group_start_s":device.commands[-1]["send_start_s"],
             "send_group_end_s":device.commands[-1]["send_end_s"],
             "max_joint_error_rad":float(np.max(np.abs(error))),"progress_rate":rate,
             "mode_feedback":int(status["mode_feedback"]),"arm_status":int(status["arm_status"])}
        for j in range(1,7):
            row[f"q{j}_rad"],row[f"q{j}_ref_rad"],row[f"u{j}_sdk_rad_s"]=float(q[j-1]),float(desired[j-1]),sent[j-1]
        records.append(row)
        rate=float(np.clip(1.-np.max(np.abs(error))/cfg["tracking_tolerance_rad"],0.,1.))
        if now-start-printed>=2:
            print(f"{step['name']}: {phase:.1f}/{times[-1]:.1f}s, max joint error={np.rad2deg(abs(error).max()):.3f}deg",flush=True)
            printed=now-start
        if phase>=times[-1] and np.max(np.abs(error))<=cfg["arrival_tolerance_rad"]:
            break
        if now-start>times[-1]*4+10:
            raise RuntimeError(f"{step['name']}: could not finish path, saved actual progress")
    # Keep the same velocity mode across task stages. The first hardware trial
    # showed a large excursion when switching CPV -> MOVE_J at the endpoint.
    device.zero(.5)


def gripper_step(device, gripper, step, cfg, rows, arm_active=True):
    device.phase=step["name"]
    start=time.monotonic()
    while time.monotonic()-start<1.5:
        if arm_active:
            _, status = device.state()
            if int(status["mode_feedback"]) == 5:
                device.velocity([0.0]*6)
        gripper.move_gripper_m(value=step["width_m"],force=step["force_n"])
        time.sleep(.05)
        value=gripper.get_gripper_status()
        if value is None or device.message_age(0x2A8)>.2:
            raise RuntimeError("Gripper feedback unavailable")
        msg=value.msg
        faults=[k for k,v in msg.foc_status.to_dict().items()
                if v and k not in ("driver_enable_status","homing_status")]
        if faults:raise RuntimeError(f"Gripper fault: {faults}")
        rows.append({"stage":step["name"],"can_timestamp_s":value.timestamp,
                     "command_width_m":step["width_m"],"width_m":msg.value,
                     "command_force_n":step["force_n"],"force_n":msg.force})
    # Width/force are observations, not a grasp classifier. A rotated cube can
    # legitimately stop wider than its side length; do not emergency-stop on it.
    print(f"{step['name']}: actual gripper width={msg.value*1000:.1f}mm force={msg.force:.2f}N",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan",type=Path,help="task.json saved by plan_pick_place.py")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument('--method',choices=('baseline','cbf','oscbf'),default='baseline')
    parser.add_argument("--record-diagnostics",action="store_true",
                        help="Record all CAN traffic, SDK states, console and three seconds after stopping")
    parser.add_argument("--speed-scale",type=float,default=1.,
                        help="Scale the saved joint velocity cap; preserve every planned path point")
    parser.add_argument("--descent-speed-scale",type=float,default=1.,
                        help="Speed scale for approach/lower near the table (default: proven original speed)")
    parser.add_argument("--start-at",help="Resume at a saved stage only if the actual state matches its start")
    parser.add_argument("--stop-after",help="For the first approach, use pregrasp and inspect the fingertip height")
    args=parser.parse_args()
    if args.method!='baseline' and args.start_at:
        parser.error('Filtered tasks require the full stage sequence to initialize held-object geometry')
    if any(not np.isfinite(v) or v<=0 for v in (args.speed_scale,args.descent_speed_scale)):
        parser.error("Speed scales must be positive finite numbers")
    task=json.loads(args.plan.read_text());cfg=task["config"]
    saved_cap=cfg["joint_velocity_limit_rad_s"]
    cfg["joint_velocity_limit_rad_s"]=saved_cap*args.speed_scale
    cfg["stage_velocity_limits_rad_s"]={name:saved_cap*args.descent_speed_scale for name in ("approach","lower")}
    task["execution_override"]={"speed_scale":args.speed_scale,
                                "descent_speed_scale":args.descent_speed_scale,
                                "saved_joint_velocity_limit_rad_s":saved_cap,'method':args.method}
    if task["report"]["status"]!="planned":raise RuntimeError("Task planning did not finish")
    steps=[s for s in task["steps"] if s["kind"] in ("arm","gripper")]
    names=[s["name"] for s in steps]
    if args.start_at:steps=steps[names.index(args.start_at):]
    if args.stop_after:steps=steps[:[s["name"] for s in steps].index(args.stop_after)+1]
    print(f"速度倍率 {args.speed_scale:g}；关节速度上限 {cfg['joint_velocity_limit_rad_s']:.3f} rad/s",flush=True)
    print(f"接近/放置下降倍率 {args.descent_speed_scale:g}；保持原规划路径",flush=True)
    for s in steps:
        if s["kind"]=="arm":
            times,q=path_arrays(s,cfg["stage_velocity_limits_rad_s"].get(s["name"],cfg["joint_velocity_limit_rad_s"]))
            print(s["name"],"minimum duration",round(times[-1],1),"s; target deg",np.round(np.rad2deg(q[-1]),2).tolist(),flush=True)
        else:print(s["name"],s["width_m"]*1000,"mm",s["force_n"],"N",flush=True)
    if not args.execute:
        print("PLAN_PREVIEW_ONLY: no CAN connection or motion",flush=True)
        return 0
    if cfg["scene_status"] not in ("measured", "operator_confirmed_approximate"):
        raise RuntimeError("Candidate scene has not been confirmed; update config and replan before motion")
    from piper_oscbf.device import PiperDevice
    out=ROOT/"results"/("pick_place_hardware_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    report={"status":"started","error":None,"controller":"MoveIt path + joint CPV tracking (not OSCBF)",
            "source_plan":str(args.plan.resolve()),"completed_stages":[],
            "speed_scale":args.speed_scale,"joint_velocity_limit_rad_s":cfg["joint_velocity_limit_rad_s"],
            "descent_speed_scale":args.descent_speed_scale,"stage_velocity_limits_rad_s":cfg["stage_velocity_limits_rad_s"],
            "object_transfer_success":"requires_operator_confirmation", "cpv_period_s": .01}
    report['method']=args.method
    report['feedback_age_clock']='monotonic callback receipt; not an end-to-end sensor latency measurement'
    if args.method!='baseline':report['controller']=f'MoveIt path + shared joint nominal + {args.method.upper()} TCP velocity safety filter'
    device=None;controls=[];grips=[];recording=None;session=None
    def terminate(*_):raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM,terminate)
    try:
        extra_sources=()
        if args.method!='baseline':
            from piper_oscbf.task_session import TaskSession, jsonable
            from piper_oscbf.task_execution import filtered_arm_step, filtered_gripper_step
            print('Prewarming complete task model and QP before hardware access.',flush=True)
            session=TaskSession(task,args.method)
            extra_sources=('piper_oscbf/task_filter.py','piper_oscbf/task_geometry.py',
                           'piper_oscbf/task_reference.py','piper_oscbf/task_session.py',
                           'piper_oscbf/task_execution.py')
        if args.record_diagnostics:
            from piper_oscbf.recording import TaskRecording, snapshot
            recording=TaskRecording(out,ROOT,extra_sources=extra_sources)
            print("DIAGNOSTIC_RESULT_DIR:",out,flush=True)
            print("执行参数：",task["execution_override"],flush=True)
        device=PiperDevice(cpv_period_s=.01);gripper=device.robot.init_effector(device.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        if recording:recording.device=device
        device.connect()
        if recording:recording.bind_connected_device(device)
        if recording:recording.mark("connected",state=snapshot(device))
        first_arm=next((s for s in steps if s["kind"]=="arm"),None)
        if first_arm is not None:
            initial,initial_status=device.state(enabled=False)
            _,q=path_arrays(first_arm,cfg["joint_velocity_limit_rad_s"])
            if np.max(np.abs(np.asarray(device.angles(initial))-q[0]))>cfg["tracking_tolerance_rad"]:
                raise RuntimeError("Actual arm does not match the planned start; no motion sent")
            if int(initial_status["mode_feedback"]) == 5:
                device.state()  # Resume only an already enabled CPV session.
                device.control_started=True
                if session is None:device.zero(.2)
            else:
                device.enable_at(device.angles(initial))
        for step in steps:
            if recording:recording.mark("stage_start",stage=step["name"],state=snapshot(device))
            if session is not None:
                if step['kind']=='arm':filtered_arm_step(device,gripper,step,cfg,controls,session,task_feedback)
                else:filtered_gripper_step(device,gripper,step,cfg,grips,session,task_feedback)
            elif step["kind"]=="arm":arm_step(device,step,cfg,controls)
            else:gripper_step(device,gripper,step,cfg,grips,arm_active=first_arm is not None)
            report["completed_stages"].append(step["name"])
            if recording:recording.mark("stage_end",stage=step["name"],state=snapshot(device))
        report["status"]="sequence_completed" if steps[-1]["name"]=="return_initial" else "staged_stop"
        if first_arm is not None:
            final,status=device.state();report["final_q_rad"]=device.angles(final)
        else:
            # Standalone gripper commands do not clear an arm estop, enable arm
            # motors, switch their mode, or send zero-velocity arm commands.
            status=device.robot.get_arm_status().msg.to_dict()
            report["final_q_rad"]=list(device.robot.get_joint_angles().msg)
        report["final_mode_feedback"]=int(status["mode_feedback"])
        report["normal_stop_action"]=("zero CPV velocity; no switch to MOVE_J" if first_arm is not None
                                     else "gripper operation only; no arm target or mode change")
        print("PIPER_TASK_FINISHED:",report["status"],report["normal_stop_action"],flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=f"{type(exc).__name__}: {exc}")
        report["exception_time_s"]=time.time()
        report["exception_monotonic_s"]=time.monotonic()
        report["traceback"]=traceback.format_exc()
        if session is not None:
            from piper_oscbf.task_session import jsonable
            if hasattr(exc,'diagnostics'):report['filter_failure']=jsonable(exc.diagnostics)
            if hasattr(exc,'state'):report['reference_failure']=jsonable(exc.state)
        if recording:
            recording.mark("exception_before_stop",error=report["error"],
                           state=snapshot(device) if device else None,traceback=report["traceback"])
        if device and device.control_started:
            if isinstance(exc,KeyboardInterrupt):
                report["stop_send_errors"]=device.emergency_stop()
            else:
                try:
                    # A zero request needs no position calculation or another
                    # read of the same state which raised the software error.
                    device.velocity([0.]*6, immediate=True)
                    time.sleep(.06)
                    stop_until=time.monotonic()+.5
                    while time.monotonic()<stop_until:
                        device.velocity([0.]*6)
                    report["stop_action"]="zero CPV requested after software error; no reset or mode switch"
                    comm_error=device.robot.get_comm_error()
                    report["stop_comm_error"]=str(comm_error) if comm_error is not None else None
                    print("已请求零CPV速度停止；SDK通信错误:",report["stop_comm_error"],
                          "；未自动切换模式或复位。",flush=True)
                except Exception as stop_error:
                    report["stop_send_error"]=str(stop_error)
                    report["stop_send_errors"]=device.emergency_stop()
        print(report["error"],flush=True)
    finally:
        if recording:
            recording.mark("passive_post_stop_started",status=report["status"],
                           state=snapshot(device) if device else None)
            print("仅继续采集停止后的状态 3 秒，不发送新的运动/复位/使能指令。",flush=True)
            try:time.sleep(3.)
            except KeyboardInterrupt:pass
            report["post_stop_state"]=snapshot(device) if device else None
            recording.finish()
            report["diagnostics"]={"raw_frames":recording.frames,"state_snapshots":recording.states,
                                   "recorder_error":recording.error,"post_stop_observation_s":3.}
        if device:
            device.disconnect();save_csv(out/"feedback.csv",device.feedback);save_csv(out/"commands.csv",device.commands)
            save_csv(out/"driver_feedback.csv",device.driver_feedback)
            report["last_driver_check"]=device.last_driver_check
            report["driver_feedback_pauses"]=device.driver_feedback_pauses
        save_csv(out/"control.csv",controls);save_csv(out/"gripper.csv",grips)
        if session is not None:
            try:report['task_filter']=session.finish(out)
            except Exception as save_error:
                report['filter_recording_error']=f'{type(save_error).__name__}: {save_error}'
        (out/"task.json").write_text(json.dumps(task,indent=2)+"\n")
        (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print("RESULT_DIR:",out,flush=True)
    return int(report["status"] in ("failed","interrupted"))


if __name__=="__main__":
    raise SystemExit(main())
