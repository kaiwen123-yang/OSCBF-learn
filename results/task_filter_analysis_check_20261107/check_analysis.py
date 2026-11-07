"""Small numerical fixtures for phase alignment, schedule identity and units.

These files are synthetic analysis inputs, not simulated or physical robot
results. No hardware process or control test framework is used.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analyze_task_filters',ROOT/'scripts/analyze_task_filters.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def fixture(name,method,elapsed,phase,*,geometry_changed=False,stale_schedule=False):
    folder=OUT/'fixtures'/name;folder.mkdir(parents=True,exist_ok=True)
    n=len(elapsed)
    summary={'status':'completed','completed_stages':['transfer'],
             'data_source':'synthetic_numeric_fixture_not_hardware',
             'object_transfer_success':'not_applicable_fixture','task_duration_s':float(elapsed[-1])}
    task={'config':{'joint_velocity_limit_rad_s':.3,'tcp_offset_m':[0,0,.1425],
                     'filter_method':method,'obstacles':[{'center_m':[0,0,0],'size_m':[.1,.1,.1]}]},
          'execution_override':{'method':method,'speed_scale':1.5},
          'steps':[{'name':'transfer','kind':'arm','trajectory':{'joint_trajectory':{
              'joint_names':[f'joint{i}' for i in range(1,7)],
              'points':[{'positions':[0.]*6,'time_from_start':{'sec':0,'nanosec':0}},
                        {'positions':[.04,0,0,0,0,0],'time_from_start':{'sec':0,'nanosec':400000000}}]}}}]}
    fingerprints={key:analysis.digest({'fixture_component':key}) for key in analysis.FINGERPRINT_KEYS}
    if geometry_changed:fingerprints['geometry']=analysis.digest({'fixture_component':'different_geometry'})
    metadata={'method':method,'fingerprints':fingerprints,'geometry_constraint_count':1,
               'constraint_labels':['obstacle_clearance','joint_lower'],
               'h_units':['m','rad'],'physical_h_units':['m','rad']}
    h=np.c_[np.asarray([.02,.01,-.001,.005,.01]),np.full(n,.5)]
    physical=h.copy();physical[:,0]+=.02
    active=np.asarray([(p>=.1 and p<=.3) for p in phase])
    nominal=np.c_[np.where(active,-.01,.01),np.full(n,.2)]
    qp=np.c_[np.full(n,.001),np.full(n,.2)]
    applied=qp.copy();applied[2,0]=-1e-7
    rows=[]
    for i,(t,p) in enumerate(zip(elapsed,phase)):
        e=.01+.02*p
        row={'stage':'transfer','time_s':t,'monotonic_s':100.+t,'reference_time_s':p,
             'phase_rate':1.,'method':method,'tcp_x_m':e,'tcp_y_m':0.,'tcp_z_m':0.,
             'tcp_ref_x_m':0.,'tcp_ref_y_m':0.,'tcp_ref_z_m':0.,
             'tcp_position_error_m':e,'orientation_error_rad':.1*p,
             'correction_joint_l2_rad_s':.01 if active[i] else 0.,'correction_task_l2':.005 if active[i] else 0.,
             'constraint_active':bool(active[i]),'geometry_active':bool(active[i]),
             'min_h_m':h[i,0],'min_physical_distance_m':physical[i,0],
             'min_applied_margin':float(applied[i].min()),'qp_solved':bool(active[i]),'qp_converged':True,
             'qp_iterations':5 if active[i] else 0,'qp_ms':.5,'sigma_min':.15,'metric_condition':140.,'feedback_age_s':.002}
        for j in range(1,7):row.update({f'q{j}_rad':0.,f'q{j}_ref_rad':0.,f'u{j}_sdk_rad_s':0.})
        rows.append(row)
    with (folder/'control.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(folder/'summary.json',summary);write_json(folder/'task.json',task)
    write_json(folder/'filter_metadata.json',metadata)
    schedule={'stage':'transfer','reference_signature':analysis.digest('fixture_reference'),
              'elapsed_s':list(elapsed),'phase_s':list(phase)}
    if stale_schedule:
        schedule['elapsed_s']=[0.,.1,.2,.3,.4];schedule['phase_s']=[0.,.1,.2,.3,.4]
    write_json(folder/'reference_schedules.json',{'stages':{'transfer':schedule}})
    np.savez_compressed(folder/'filter_constraints.npz',h=h,physical_h=physical,
                        nominal_margin=nominal,qp_margin=qp,applied_margin=applied)
    return folder


def main():
    base=fixture('cbf_regular','cbf',[0.,.1,.2,.3,.4],[0.,.1,.2,.3,.4])
    same=fixture('oscbf_same_schedule','oscbf',[0.,.1,.2,.3,.4],[0.,.1,.2,.3,.4])
    slow=fixture('oscbf_adaptive_slow','oscbf',[0.,.2,.4,.6,.8],[0.,.05,.1,.25,.4])
    stale=fixture('oscbf_stale_declared_schedule','oscbf',[0.,.2,.4,.6,.8],[0.,.05,.1,.25,.4],stale_schedule=True)
    changed=fixture('oscbf_changed_geometry','oscbf',[0.,.1,.2,.3,.4],[0.,.1,.2,.3,.4],geometry_changed=True)
    reports={}
    for name,other in [('same',same),('different',slow),('stale',stale),('changed_geometry',changed)]:
        reports[name]=analysis.analyze_paths(base,other,OUT/'reports'/name)
    assert reports['same']['comparison']['common_conditions_verified']
    assert reports['same']['comparison']['reference_timing']['strict_same_schedule']
    for name in ('different','stale'):
        assert not reports[name]['comparison']['reference_timing']['strict_same_schedule']
    assert not reports['changed_geometry']['comparison']['common_conditions_verified']
    window=reports['different']['comparison']['phase_aligned_errors']['transfer']['windows']['union_constraint_active']
    assert abs(window['position']['run_a']['phase_weighted_rms']-window['position']['run_b']['phase_weighted_rms'])<1e-14
    assert abs(window['orientation']['run_a']['phase_weighted_rms']-window['orientation']['run_b']['phase_weighted_rms'])<1e-14
    for report in reports.values():
        assert not report['runs'][0]['operator_confirmed_transfer']
        assert report['runs'][0]['can']['error_frames'] is None
        assert report['runs'][0]['arrays']['alignment_valid']
    negatives=list(csv.DictReader((OUT/'reports/different/run_a_negative_values.csv').open()))
    assert any(r['array']=='h' and r['raw_value']=='-0.001' and r['unit']=='m' for r in negatives)
    assert any(r['array']=='applied_margin' and float(r['raw_value'])==-1e-7 for r in negatives)
    old_hash=analysis.digest({'summary':json.loads((base/'summary.json').read_text()),'control':(base/'control.csv').read_text()})
    try:
        analysis.analyze_paths(base,None,base/'analysis')
    except ValueError:pass
    else:raise AssertionError('Analyzer accepted an output inside input data')
    assert old_hash==analysis.digest({'summary':json.loads((base/'summary.json').read_text()),'control':(base/'control.csv').read_text()})
    result={'status':'passed','scope':'tiny numerical analysis fixtures; no hardware or robot simulation',
        'identical_actual_schedule_detected':True,'different_actual_schedule_detected':True,
        'stale_identical_declared_schedule_does_not_override_control_trace':True,
        'method_independent_geometry_mismatch_detected':True,
        'phase_aligned_errors_equal_for_same_error_as_function_of_phase':True,
        'raw_negative_h_and_small_negative_applied_margin_preserved':True,
        'missing_CAN_not_reported_as_zero_errors':True,
        'program_completion_not_operator_confirmation':True,
        'output_inside_input_rejected':True,
        'phase_aligned_fixture_window':window}
    write_json(OUT/'checks.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
