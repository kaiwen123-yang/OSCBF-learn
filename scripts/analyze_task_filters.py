"""Text/JSON analysis of complete-task CBF/OSCBF records, without hardware.

Inputs are never changed. Output must be a dedicated directory outside the input
run directories. Pair error comparisons use intersecting per-stage reference
phase domains; the same controller code does not imply the same time schedule.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

METHOD_KEYS = {'method', 'filter_method', 'safety_filter_method'}
FINGERPRINT_KEYS = {
    'path': ('path', 'task_path', 'reference_path', 'plan'),
    'geometry': ('geometry', 'scene_geometry', 'geometry_config', 'scene'),
    'parameters': ('parameters', 'runtime_parameters', 'config'),
    'controller': ('controller', 'controller_code', 'nominal_controller'),
    'constraints': ('constraints', 'constraint_config', 'constraints_code'),
    'model': ('model', 'model_urdf', 'model_sha256'),
}
ARRAY_NAMES = ('h', 'physical_h', 'nominal_margin', 'qp_margin', 'applied_margin')
FILTER_COMMON_FIELDS = ('frame','task_frame','nominal','kp_joint','task_weights','objective_scale',
    'rank_policy','alpha_geometry','alpha_joint','velocity_quantization_rad_s','quantization_policy',
    'nominal_fast_path','solver_tol','residual_tol','physical_tolerance_m','solver_max_iter',
    'geometry_constraint_count','constraint_labels','h_row_semantics','physical_h_semantics','backend')


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def number(value):
    try: return float(value)
    except (TypeError, ValueError): return math.nan


def boolean(value):
    if value is True or value == 1: return 1.
    if value is False or value == 0: return 0.
    text = str(value).strip().lower()
    if text in ('true', '1'): return 1.
    if text in ('false', '0'): return 0.
    return math.nan


def series(rows, name, cast=number):
    return np.asarray([cast(r.get(name)) for r in rows], dtype=float)


def stats(values, weights=None):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = {'recorded_samples': int(valid.sum()), 'missing_or_nonfinite_samples': int((~valid).sum())}
    if not valid.any():
        return {**result, 'min': None, 'median': None, 'p95': None, 'p99': None, 'max': None,
                'sample_rms': None, 'time_weighted_rms': None}
    a = values[valid]
    result.update(min=float(a.min()), median=float(np.median(a)), p95=float(np.percentile(a,95)),
                  p99=float(np.percentile(a,99)), max=float(a.max()), sample_rms=float(np.sqrt(np.mean(a*a))))
    if weights is not None:
        w = np.asarray(weights)[valid]
        result['time_weighted_rms'] = float(np.sqrt(np.sum(w*a*a)/np.sum(w))) if np.sum(w)>0 else None
    return result


def without_method(value):
    if isinstance(value, dict): return {k:without_method(v) for k,v in value.items() if k not in METHOD_KEYS}
    if isinstance(value, list): return [without_method(v) for v in value]
    return value


def task_fingerprints(task):
    if not isinstance(task,dict): return {'path':None,'parameters':None}
    steps = task.get('steps')
    return {'path':digest(steps) if isinstance(steps,list) else None,
            'parameters':digest(without_method({'config':task.get('config'),
                'execution_override':task.get('execution_override')})) if 'config' in task else None}


def metadata_fingerprints(metadata):
    source = metadata.get('fingerprints', {})
    output = {}
    for key, aliases in FINGERPRINT_KEYS.items():
        output[key] = next((source[a] for a in aliases if source.get(a)), None)
    return output


def interval_weights(rows):
    times = series(rows,'monotonic_s')
    weights = np.zeros(len(rows))
    invalid = []
    for i in range(max(0,len(rows)-1)):
        if rows[i].get('stage') != rows[i+1].get('stage'): continue
        delta = times[i+1]-times[i]
        if np.isfinite(delta) and delta>0:
            weights[i] = delta
        else:
            invalid.append(i)
    return times, weights, invalid


def active_summary(flags, weights, rows):
    valid, active = np.isfinite(flags), flags==1
    known_time = float(weights[valid].sum())
    return {'known_samples':int(valid.sum()), 'unknown_samples':int((~valid).sum()),
            'active_samples':int(active.sum()),
            'active_fraction_of_known_samples':float(active.sum()/valid.sum()) if valid.any() else None,
            'active_observed_time_s':float(weights[active].sum()),
            'known_observed_time_s':known_time,
            'active_fraction_of_known_time':float(weights[active].sum()/known_time) if known_time>0 else None,
            'active_tcp_position_error_m':stats(series(rows,'tcp_position_error_m')[active],weights[active]),
            'active_orientation_error_rad':stats(series(rows,'orientation_error_rad')[active],weights[active])}


def column_labels(metadata, array_name, width):
    keys = ('h_labels','constraint_labels') if array_name in ('h','physical_h') else ('constraint_labels',)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value,(list,tuple)) and len(value)==width:
            return [str(v.get('name',v.get('label',v))) if isinstance(v,dict) else str(v) for v in value], True
    return [f'{array_name}[{i}]' for i in range(width)], False


def column_units(metadata, array_name, width):
    specific = metadata.get(f'{array_name}_units')
    if isinstance(specific,list) and len(specific)==width: return specific
    if 'geometry_constraint_count' not in metadata: return ['unspecified']*width
    geometry = int(metadata.get('geometry_constraint_count',0))
    if array_name in ('h','physical_h'):
        if width == geometry+24: return ['m']*geometry+['rad']*12+['rad/s']*12
        return ['m' if i<geometry else 'unspecified' for i in range(width)]
    return ['m/s' if i<geometry else 'rad/s' for i in range(width)]


def negative_episodes(mask, rows, times, weights):
    groups=[]
    for i in np.flatnonzero(mask):
        i=int(i)
        if not groups or i!=groups[-1][-1]+1 or rows[i].get('stage')!=rows[groups[-1][-1]].get('stage'):
            groups.append([])
        groups[-1].append(i)
    return [{'stage':rows[g[0]].get('stage'), 'first_sample':g[0], 'last_sample':g[-1],
             'first_monotonic_s':times[g[0]], 'last_monotonic_s':times[g[-1]],
             'sample_hold_duration_s':float(weights[g].sum())} for g in groups]


def array_analysis(path, metadata, rows, times, weights, out, label):
    source=path/'filter_constraints.npz'
    result={'available':source.exists(),'alignment_valid':True,'validation_errors':[], 'arrays':{}}
    if not source.exists():
        result['alignment_valid']=False
        result['validation_errors'].append('filter_constraints.npz missing; per-constraint negatives unavailable')
        fields=['sample_index','stage','monotonic_s','stage_elapsed_s','reference_phase_s','array',
                'constraint_index','constraint_label','unit','raw_value','sample_hold_interval_s']
        with (out/f'{label}_negative_values.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for name,unit in [('min_h_m','m'),('min_physical_distance_m','m'),('min_applied_margin','mixed')]:
                values=series(rows,name)
                for i in np.flatnonzero(np.isfinite(values)&(values<0)):
                    writer.writerow({'sample_index':int(i),'stage':rows[i].get('stage'),
                        'monotonic_s':times[i],'stage_elapsed_s':number(rows[i].get('time_s')),
                        'reference_phase_s':number(rows[i].get('reference_time_s')),
                        'array':'scalar_minimum_fallback','constraint_index':'unknown',
                        'constraint_label':name,'unit':unit,'raw_value':values[i],
                        'sample_hold_interval_s':weights[i]})
        result['negative_value_scope']='Only recorded scalar minima; original per-constraint arrays absent'
        return result
    arrays={}
    with np.load(source,allow_pickle=False) as archive:
        for name in ARRAY_NAMES:
            if name not in archive:
                result['validation_errors'].append(f'{name} missing from NPZ');result['alignment_valid']=False;continue
            value=np.asarray(archive[name],dtype=float)
            if value.ndim!=2 or value.shape[0]!=len(rows):
                result['validation_errors'].append(f'{name} shape {value.shape} does not align with {len(rows)} control rows')
                result['alignment_valid']=False;continue
            arrays[name]=value
    fields=['sample_index','stage','monotonic_s','stage_elapsed_s','reference_phase_s','array',
            'constraint_index','constraint_label','unit','raw_value','sample_hold_interval_s']
    with (out/f'{label}_negative_values.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,value in arrays.items():
            labels,labelled=column_labels(metadata,name,value.shape[1]);units=column_units(metadata,name,value.shape[1])
            if not labelled:
                result['alignment_valid']=False;result['validation_errors'].append(f'{name} labels missing or wrong width')
            columns=[]
            for k in range(value.shape[1]):
                negative=np.isfinite(value[:,k])&(value[:,k]<0)
                columns.append({'index':k,'label':labels[k],'unit':units[k],
                    'values':stats(value[:,k]),'strictly_negative_samples':int(negative.sum()),
                    'strictly_negative_sample_hold_duration_s':float(weights[negative].sum()),
                    'negative_episodes':negative_episodes(negative,rows,times,weights)})
                for i in np.flatnonzero(negative):
                    writer.writerow({'sample_index':int(i),'stage':rows[i].get('stage'),
                        'monotonic_s':times[i],'stage_elapsed_s':number(rows[i].get('time_s')),
                        'reference_phase_s':number(rows[i].get('reference_time_s')),
                        'array':name,'constraint_index':k,'constraint_label':labels[k],
                        'unit':units[k],'raw_value':value[i,k],'sample_hold_interval_s':weights[i]})
            result['arrays'][name]={'shape':list(value.shape),'columns':columns}
    geometry=int(metadata.get('geometry_constraint_count',0))
    for name,csv_field in [('h','min_h_m'),('physical_h','min_physical_distance_m'),('applied_margin','min_applied_margin')]:
        if name not in arrays: continue
        values=arrays[name]
        if name!='applied_margin':
            if not 0<geometry<=values.shape[1]: continue
            values=values[:,:geometry]
        if values.shape[1]==0: continue
        derived=np.min(values,axis=1)
        supplied=series(rows,csv_field)
        valid=np.isfinite(derived)&np.isfinite(supplied)
        max_diff=float(np.max(np.abs(derived[valid]-supplied[valid]))) if valid.any() else None
        result[f'{csv_field}_npz_csv_max_difference']=max_diff
        if max_diff is not None and max_diff>1e-8:
            result['alignment_valid']=False;result['validation_errors'].append(f'{name} and {csv_field} disagree by {max_diff}')
        result[f'{csv_field}_from_npz']=stats(derived,weights)
    return result


def normalize_schedules(source, rows):
    candidates=source
    if isinstance(source,dict): candidates=source.get('stages',source.get('schedules',source))
    if isinstance(candidates,list): candidates={str(v.get('stage')):v for v in candidates if isinstance(v,dict)}
    output={}
    for stage in dict.fromkeys(r.get('stage','unknown') for r in rows):
        entry=candidates.get(stage) if isinstance(candidates,dict) else None
        c=[r for r in rows if r.get('stage','unknown')==stage]
        elapsed,phase=series(c,'time_s'),series(c,'reference_time_s')
        signature=entry.get('reference_signature',entry.get('signature')) if isinstance(entry,dict) else None
        trace_match=None
        if isinstance(entry,dict) and isinstance(entry.get('phase_s'),list) and len(entry['phase_s'])==len(phase):
            trace_match=bool(np.allclose(np.asarray(entry['phase_s'],float),phase,atol=1e-10,rtol=0))
        valid=(len(elapsed)>0 and len(elapsed)==len(phase) and np.all(np.isfinite(elapsed))
               and np.all(np.isfinite(phase)) and np.all(np.diff(elapsed)>0) and np.all(np.diff(phase)>=0))
        output[stage]={'source':'control.csv_actual_execution','reference_signature':signature,'valid':bool(valid),
                       'schedule_metadata_recorded':isinstance(entry,dict),
                       'metadata_phase_sequence_matches_control':trace_match,
                       'elapsed_s':elapsed.tolist(),'phase_s':phase.tolist()}
    return output


def load_run(path,out,label):
    path=path.resolve()
    summary=read_json(path/'summary.json',{})
    metadata=read_json(path/'filter_metadata.json',{})
    task=read_json(path/'task.json',{})
    if (path/'control.csv').exists():
        with (path/'control.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    else: rows=[]
    events=[]
    if (path/'events.jsonl').exists():
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line.strip()]
    times,weights,invalid_intervals=interval_weights(rows)
    active=series(rows,'constraint_active',boolean);geometry_active=series(rows,'geometry_active',boolean)
    correction=series(rows,'correction_joint_l2_rad_s')
    qp_solved=series(rows,'qp_solved',boolean);qp_converged=series(rows,'qp_converged',boolean)
    status=summary.get('status','unknown')
    methods=sorted({r['method'] for r in rows if r.get('method')})
    method=methods[0] if len(methods)==1 else metadata.get('method',summary.get('method'))
    operator=summary.get('operator_confirmation')
    transfer=summary.get('object_transfer_success',summary.get('physical_transfer_success'))
    operator_confirmed=bool(transfer=='operator_confirmed' or (isinstance(operator,dict) and operator.get('confirmed') is True))
    starts=[e['monotonic_s'] for e in events if e.get('event')=='stage_start' and 'monotonic_s' in e]
    ends=[e['monotonic_s'] for e in events if e.get('event')=='stage_end' and 'monotonic_s' in e]
    origin=min(starts) if starts else (float(times[0]-number(rows[0].get('time_s'))) if len(rows) else None)
    completed_duration=max(ends)-min(starts) if starts and ends else None
    schedules=normalize_schedules(read_json(path/'reference_schedules.json',{}),rows)
    for stage,s in schedules.items():
        c=[r for r in rows if r.get('stage')==stage]
        offsets=series(c,'monotonic_s')-series(c,'time_s')
        s['stage_offset_from_task_start_s']=float(np.median(offsets)-origin) if len(c) and origin is not None else None
    arrays=array_analysis(path,metadata,rows,times,weights,out,label)
    can={'raw_available':(path/'raw_can.csv').exists(),'raw_frames':None,'error_frames':None,'error_phases':None,'bus_off_error_frames':None}
    if can['raw_available']:
        count=error_count=busoff=0;phases=Counter()
        with (path/'raw_can.csv').open() as f:
            for row in csv.DictReader(f):
                count+=1
                if boolean(row.get('is_error'))==1:
                    error_count+=1;phases[row.get('phase','unknown')]+=1
                    busoff+=bool(int(row['arbitration_id'],16)&0x40)
        can.update(raw_frames=count,error_frames=error_count,error_phases=dict(phases),bus_off_error_frames=busoff)
    recording=read_json(path/'recording.json',{})
    can['interface_delta']=recording.get('interface_delta')
    can['recorder_error']=recording.get('recorder_error')
    can['sdk_send_calls']=recording.get('sdk_send_calls')
    raw_sources=['summary.json','filter_metadata.json','task.json','control.csv','filter_constraints.npz',
                 'reference_schedules.json','recording.json','raw_can.csv']
    common_filter_parameters={k:metadata[k] for k in FILTER_COMMON_FIELDS if k in metadata}
    result={'source_directory':str(path),'label':label,'method':method,'methods_in_control':methods,
        'data_source':summary.get('data_source',metadata.get('data_source','not_recorded')),
        'status':status,'execution_completed':status in ('sequence_completed','completed'),
        'completed_stages':summary.get('completed_stages',[]),'error':summary.get('error'),
        'operator_confirmed_transfer':operator_confirmed,'transfer_status_raw':transfer,'operator_confirmation_raw':operator,
        'task_duration_s':summary.get('task_duration_s',completed_duration),
        'task_duration_definition':'Explicit summary task duration, else first stage_start to last stage_end; partial runs list only completed-stage span.',
        'control_samples':len(rows),'observed_control_interval_time_s':float(weights.sum()),
        'interval_definition':'Previous-sample hold over consecutive same-stage control rows; stage gaps and final rows have zero weight.',
        'invalid_control_interval_indices':invalid_intervals,
        'constraint_activation':active_summary(active,weights,rows),
        'geometry_activation':active_summary(geometry_active,weights,rows),
        'correction_joint_l2_rad_s':stats(correction,weights),
        'correction_task_l2':stats(series(rows,'correction_task_l2'),weights),
        'nonzero_joint_correction_samples_above_1e_9':int((correction>1e-9).sum()),
        'geometry_intervention_with_nonzero_correction_samples':int(((geometry_active==1)&(correction>1e-9)).sum()),
        'scalar_min_h_m':stats(series(rows,'min_h_m'),weights),
        'scalar_min_physical_distance_m':stats(series(rows,'min_physical_distance_m'),weights),
        'scalar_min_applied_margin_mixed_units':stats(series(rows,'min_applied_margin'),weights),
        'control_cycle_ms':stats(weights[weights>0]*1000),
        'feedback_age_ms':stats(series(rows,'feedback_age_s')*1000),
        'qp_ms':stats(series(rows,'qp_ms')),'qp_iterations':stats(series(rows,'qp_iterations')),
        'qp_ms_on_solved_samples':stats(series(rows,'qp_ms')[qp_solved==1]),
        'compute_time_scope':metadata.get('timing_scope','not_recorded'),
        'qp_solved_true_samples':int((qp_solved==1).sum()),
        'qp_converged_false_samples':int((qp_converged==0).sum()),
        'qp_convergence_unknown_samples':int((~np.isfinite(qp_converged)).sum()),
        'sigma_min':stats(series(rows,'sigma_min')),'metric_condition':stats(series(rows,'metric_condition')),
        'arrays':arrays,'can':can,'fingerprints':metadata_fingerprints(metadata),
        'derived_task_fingerprints':{**task_fingerprints(task),
            'filter_parameters':digest(common_filter_parameters) if common_filter_parameters else None},
        'common_filter_parameters':common_filter_parameters,'reference_schedules':schedules,
        'negative_values_file':f'{label}_negative_values.csv',
        'source_sha256':{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in raw_sources if (path/name).exists()},
        'measurement_scope':'TCP positions, orientation errors, geometry distances and h are model/FK-derived unless an independent measurement source is explicitly recorded. Operator transfer confirmation is separate.'}
    if not np.isfinite(geometry_active).all() or not len(rows):
        result['intervention_scope']='unknown_or_incomplete_activation_record'
    elif result['geometry_intervention_with_nonzero_correction_samples']==0:
        result['intervention_scope']='task_integration_without_observed_geometry_filter_intervention'
    else: result['intervention_scope']='geometry_filter_intervention_observed'
    result['invalid_reference_schedule_stages']=[stage for stage,s in schedules.items() if not s['valid']]
    result['analysis_input_valid']=bool(not invalid_intervals and len(methods)<=1 and arrays['alignment_valid']
                                       and not result['invalid_reference_schedule_stages'])
    return result,rows


def schedule_comparison(a,b,phase_tolerance,elapsed_tolerance):
    sa,sb=a['reference_schedules'],b['reference_schedules'];details={}
    overall=bool(sa) and set(sa)==set(sb) and a['execution_completed'] and b['execution_completed']
    for stage in sorted(set(sa)|set(sb)):
        x,y=sa.get(stage),sb.get(stage)
        if not x or not y or not x['valid'] or not y['valid']:
            details[stage]={'same':False,'reason':'missing_stage_or_invalid_trace'};overall=False;continue
        ta,tb=np.asarray(x['elapsed_s']),np.asarray(y['elapsed_s'])
        pa,pb=np.asarray(x['phase_s']),np.asarray(y['phase_s'])
        lo,hi=max(ta[0],tb[0]),min(ta[-1],tb[-1])
        overlap=hi>=lo
        grid=np.unique(np.r_[ta[(ta>=lo)&(ta<=hi)],tb[(tb>=lo)&(tb<=hi)],lo,hi]) if overlap else np.zeros(0)
        difference=float(np.max(np.abs(np.interp(grid,ta,pa)-np.interp(grid,tb,pb)))) if len(grid) else None
        signature_equal=bool(x['reference_signature']) and x['reference_signature']==y['reference_signature']
        offset_delta=(abs(x['stage_offset_from_task_start_s']-y['stage_offset_from_task_start_s'])
            if x['stage_offset_from_task_start_s'] is not None and y['stage_offset_from_task_start_s'] is not None else None)
        same=bool(signature_equal and x['schedule_metadata_recorded'] and y['schedule_metadata_recorded']
            and x['metadata_phase_sequence_matches_control'] is not False and y['metadata_phase_sequence_matches_control'] is not False
            and abs(ta[0]-tb[0])<=elapsed_tolerance and abs(ta[-1]-tb[-1])<=elapsed_tolerance
            and abs(pa[0]-pb[0])<=phase_tolerance and abs(pa[-1]-pb[-1])<=phase_tolerance
            and difference is not None and difference<=phase_tolerance and offset_delta is not None and offset_delta<=elapsed_tolerance)
        details[stage]={'same':same,'reference_signature_equal':signature_equal,
            'elapsed_domains_s':[[ta[0],ta[-1]],[tb[0],tb[-1]]], 'phase_domains_s':[[pa[0],pa[-1]],[pb[0],pb[-1]]],
            'max_phase_difference_at_common_elapsed_s':difference,'stage_start_offset_difference_s':offset_delta}
        overall=overall and same
    return {'strict_same_schedule':overall,'scope':'Actual recorded control stages of two completed executions; partial runs cannot earn this label.',
            'phase_tolerance_s':phase_tolerance,'elapsed_tolerance_s':elapsed_tolerance,
            'comparison':'Piecewise-linear phase(elapsed) on union of knots, equal domains/signatures, and stage offsets from task start.','stages':details}


def phase_curve(rows,stage):
    c=[r for r in rows if r.get('stage')==stage]
    p=series(c,'reference_time_s')
    if len(c)<2 or not np.isfinite(p).all() or np.any(np.diff(p)<0): return None
    # Use the last observed state at a repeated phase. All paused-time extremes
    # remain in per-run metrics; preserve them explicitly here as well.
    unique,first,counts=np.unique(p,return_index=True,return_counts=True)
    last=first+counts-1
    data={'phase':unique,'position':series(c,'tcp_position_error_m')[last],
          'orientation':series(c,'orientation_error_rad')[last]}
    for name in ('constraint_active','geometry_active'):
        flags=series(c,name,boolean)
        data[name]=np.asarray([float(np.max(flags[i:i+n])) if np.isfinite(flags[i:i+n]).all() else math.nan
                               for i,n in zip(first,counts)])
    repeated=np.repeat(counts>1,counts)
    _,weights,_=interval_weights(c)
    data['pause_summary']={'repeated_phase_samples':int(repeated.sum()),
        'repeated_phase_sample_hold_time_s':float(weights[repeated].sum()),
        'repeated_phase_tcp_position_error_m':stats(series(c,'tcp_position_error_m')[repeated],weights[repeated]),
        'repeated_phase_orientation_error_rad':stats(series(c,'orientation_error_rad')[repeated],weights[repeated])}
    return data


def phase_aligned_comparison(rows_a,rows_b,max_step):
    output={}
    stages=sorted({r.get('stage') for r in rows_a}&{r.get('stage') for r in rows_b})
    for stage in stages:
        a,b=phase_curve(rows_a,stage),phase_curve(rows_b,stage)
        if a is None or b is None:
            output[stage]={'available':False,'reason':'insufficient_or_nonmonotone_phase'};continue
        low,high=max(a['phase'][0],b['phase'][0]),min(a['phase'][-1],b['phase'][-1])
        if high<=low:
            output[stage]={'available':False,'reason':'zero_length_intersection_or_hold','run_a_pause':a['pause_summary'],'run_b_pause':b['pause_summary']};continue
        n=max(2,math.ceil((high-low)/max_step)+1)
        grid=np.linspace(low,high,n);d=np.diff(grid);weights=np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]
        errors={}
        flags={}
        for label,c in [('run_a',a),('run_b',b)]:
            errors[label]={field:np.interp(grid,c['phase'],c[field]) for field in ('position','orientation')}
            idx=np.clip(np.searchsorted(c['phase'],grid,side='right')-1,0,len(c['phase'])-1)
            flags[label]={field:c[field][idx] for field in ('constraint_active','geometry_active')}
        masks={'whole_common_phase':np.ones(n,dtype=bool)}
        for field in ('constraint_active','geometry_active'):
            known=np.isfinite(flags['run_a'][field])&np.isfinite(flags['run_b'][field])
            masks[f'union_{field}']=known&((flags['run_a'][field]==1)|(flags['run_b'][field]==1))
            masks[f'intersection_{field}']=known&((flags['run_a'][field]==1)&(flags['run_b'][field]==1))
        windows={}
        for window,mask in masks.items():
            item={'phase_weight_s':float(weights[mask].sum()),'phase_fraction':float(weights[mask].sum()/(high-low))}
            for field in ('position','orientation'):
                usable=mask&np.isfinite(errors['run_a'][field])&np.isfinite(errors['run_b'][field])
                den=float(weights[usable].sum())
                item[field]={'common_valid_phase_weight_s':den}
                for label in ('run_a','run_b'):
                    v=errors[label][field]
                    item[field][label]={'phase_weighted_rms':float(np.sqrt(np.sum(weights[usable]*v[usable]**2)/den)) if den>0 else None,
                                       'max':float(np.max(v[usable])) if usable.any() else None}
            windows[window]=item
        output[stage]={'available':True,'phase_domain_s':[low,high],'grid_points':n,
            'alignment_rule':'Uniform common-phase grid; error interpolation after taking last sample at duplicate phase; active flag is any recorded activation at that phase. Pause extremes/durations reported separately.',
            'position_unit':'m','orientation_unit':'rad','windows':windows,
            'run_a_pause':a['pause_summary'],'run_b_pause':b['pause_summary']}
    return output


def compare_runs(a,rows_a,b,rows_b,args):
    fingerprints={}
    for key in FINGERPRINT_KEYS:
        x,y=a['fingerprints'][key],b['fingerprints'][key]
        fingerprints[key]={'run_a':x,'run_b':y,'both_recorded':bool(x) and bool(y),'equal':bool(x) and bool(y) and x==y}
    derived={key:{'run_a':a['derived_task_fingerprints'][key],'run_b':b['derived_task_fingerprints'][key],
                  'equal':bool(a['derived_task_fingerprints'][key]) and a['derived_task_fingerprints'][key]==b['derived_task_fingerprints'][key]}
             for key in ('path','parameters','filter_parameters')}
    equal=all(v['equal'] for v in fingerprints.values()) and all(v['equal'] for v in derived.values())
    schedules=schedule_comparison(a,b,args.phase_tolerance_s,args.elapsed_tolerance_s)
    return {'run_a_method':a['method'],'run_b_method':b['method'],
        'cbf_oscbf_pair':{a['method'],b['method']}=={'cbf','oscbf'},
        'common_conditions_verified':equal,
        'both_analysis_inputs_valid':a['analysis_input_valid'] and b['analysis_input_valid'],
        'recorded_fingerprints':fingerprints,'recomputed_task_fingerprints':derived,
        'fingerprint_basis':'Recorded method-independent geometry/controller/constraint/model provenance; actual saved steps and executed config/overrides are independently hashed after removing only explicit method selectors.',
        'both_execution_completed':a['execution_completed'] and b['execution_completed'],
        'both_operator_confirmed_transfer':a['operator_confirmed_transfer'] and b['operator_confirmed_transfer'],
        'task_durations_s':{'run_a':a['task_duration_s'],'run_b':b['task_duration_s']},
        'reference_timing':schedules,
        'phase_aligned_errors':phase_aligned_comparison(rows_a,rows_b,args.phase_grid_step_s),
        'interpretation':'One pair supports only this recorded pair. No generalized or statistical superiority is inferred. If common conditions are unverified, aligned numbers are descriptive diagnostics only. Per-run own-active-window RMS is not used as an unaligned pair performance comparison.'}


def render_report(runs,comparison):
    lines=['# 完整任务安全过滤记录分析','',
           'TCP 轨迹、误差、距离及 h 按模型/FK 来源解释；程序完成与现场物体转移确认分别列出。未生成图形。','']
    for r in runs:
        lines += [f"## {r['label']}：{r['method']}",'',
            f"状态 `{r['status']}`；程序完成：{r['execution_completed']}；现场确认物体转移：{r['operator_confirmed_transfer']}。",
            f"完成阶段：{', '.join(r['completed_stages']) or '未记录'}。任务耗时：{r['task_duration_s']} s。来源：`{r['data_source']}`。",'',
            f"控制样本 {r['control_samples']}，可计权控制区间 {r['observed_control_interval_time_s']:.6f} s；不跨阶段拼接时间，不将未记录的停止保持时间计入最后一个控制样本。",
            f"约束介入样本比例：{r['constraint_activation']['active_fraction_of_known_samples']}；几何约束介入样本比例：{r['geometry_activation']['active_fraction_of_known_samples']}。",
            f"几何介入且速度修改量 >1e-9 的样本数：{r['geometry_intervention_with_nonzero_correction_samples']}。解释范围：`{r['intervention_scope']}`。",
            f"几何 h 最小记录值：{r['scalar_min_h_m']['min']} m；物理几何间隙最小记录值：{r['scalar_min_physical_distance_m']['min']} m。",
            f"QP 未收敛记录样本：{r['qp_converged_false_samples']}；收敛状态缺失样本：{r['qp_convergence_unknown_samples']}。",
            f"原始 CAN 错误帧：{r['can']['error_frames']}；bus-off 错误帧：{r['can']['bus_off_error_frames']}。缺少原始数据时不会写成零错误。",'',
            f"数组与标签对齐有效：{r['arrays']['alignment_valid']}。所有严格负值及其样本保持时间见 `{r['negative_values_file']}`，不删除小负值；逐约束单位、负值区间和运行时统计见 JSON。",'']
    if comparison:
        lines += ['## 配对条件与时间轴','',
            f"方法为 CBF/OSCBF 配对：{comparison['cbf_oscbf_pair']}；共同条件指纹已核对一致：{comparison['common_conditions_verified']}。",
            f"实际参考时间日程严格一致：{comparison['reference_timing']['strict_same_schedule']}。",
            f"判据包含参考签名、完整时间/相位区间、分段函数 phase(elapsed) 及阶段相对任务起点的偏移；相位容差 {comparison['reference_timing']['phase_tolerance_s']} s，时间容差 {comparison['reference_timing']['elapsed_tolerance_s']} s。",'',
            '配对误差采用每个阶段共同相位区间。重复相位取最后一个样本进行相位映射，同时单独保留暂停期间的最大误差与持续时间。介入期对比用双方共同的介入相位窗口，不直接比较两个未对齐时间窗口的 RMS。','',
            '| 阶段 | 共同约束介入相位权重（s） | A TCP 相位 RMS（m） | B TCP 相位 RMS（m） | A 姿态相位 RMS（rad） | B 姿态相位 RMS（rad） |',
            '|---|---:|---:|---:|---:|---:|']
        for stage,d in comparison['phase_aligned_errors'].items():
            if not d['available']: continue
            w=d['windows']['union_constraint_active']
            lines.append(f"| {stage} | {w['phase_weight_s']} | {w['position']['run_a']['phase_weighted_rms']} | {w['position']['run_b']['phase_weighted_rms']} | {w['orientation']['run_a']['phase_weighted_rms']} | {w['orientation']['run_b']['phase_weighted_rms']} |")
        lines += ['', '一组配对只支持该次运行；没有统计显著性或一般性优势结论。指纹缺失/不同、任务未完成或现场未确认时，相关缺口原样保留。','']
    return '\n'.join(lines)


def analyze_paths(run,compare,out,*,phase_tolerance_s=1e-6,elapsed_tolerance_s=1e-6,phase_grid_step_s=.01):
    inputs=[Path(run).resolve()]+([Path(compare).resolve()] if compare else [])
    out=Path(out).resolve()
    if any(out==p or p in out.parents for p in inputs):
        raise ValueError('Output must be outside every input run directory')
    if any(not p.is_dir() for p in inputs): raise FileNotFoundError('An input run directory is missing')
    expected=[str(p) for p in inputs]
    marker=out/'analysis_manifest.json'
    if out.exists() and any(out.iterdir()):
        prior=read_json(marker,{})
        if prior.get('tool')!='analyze_task_filters' or prior.get('inputs')!=expected:
            raise ValueError('Nonempty output is not a prior analysis of these same inputs')
    out.mkdir(parents=True,exist_ok=True)
    write_json(marker,{'tool':'analyze_task_filters','schema':1,'inputs':expected})
    loaded=[load_run(p,out,f'run_{chr(97+i)}') for i,p in enumerate(inputs)]
    args=argparse.Namespace(phase_tolerance_s=phase_tolerance_s,elapsed_tolerance_s=elapsed_tolerance_s,phase_grid_step_s=phase_grid_step_s)
    comparison=compare_runs(*loaded[0],*loaded[1],args) if len(loaded)==2 else None
    report={'schema':1,'runs':[r for r,_ in loaded],'comparison':comparison,
        'negative_duration_definition':'Strict h/margin<0 at recorded samples, zero-order hold until next same-stage control sample; no cross-stage gap extrapolation. All raw negative values are retained in CSV.',
        'strict_schedule_is_not_performance_superiority':True}
    write_json(out/'analysis.json',report)
    (out/'ANALYSIS.md').write_text(render_report(report['runs'],comparison),encoding='utf-8')
    return clean_json(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase-tolerance-s',type=float,default=1e-6)
    parser.add_argument('--elapsed-tolerance-s',type=float,default=1e-6)
    parser.add_argument('--phase-grid-step-s',type=float,default=.01)
    args=parser.parse_args()
    if min(args.phase_tolerance_s,args.elapsed_tolerance_s,args.phase_grid_step_s)<=0:
        parser.error('Analysis tolerances and grid step must be positive')
    report=analyze_paths(args.run,args.compare,args.out,phase_tolerance_s=args.phase_tolerance_s,
        elapsed_tolerance_s=args.elapsed_tolerance_s,phase_grid_step_s=args.phase_grid_step_s)
    print(json.dumps({'output':str(args.out.resolve()),'runs':[{'method':r['method'],'status':r['status'],
        'operator_confirmed_transfer':r['operator_confirmed_transfer'],'input_valid':r['analysis_input_valid']} for r in report['runs']],
        'strict_same_schedule':report['comparison']['reference_timing']['strict_same_schedule'] if report['comparison'] else None,
        'common_conditions_verified':report['comparison']['common_conditions_verified'] if report['comparison'] else None},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
