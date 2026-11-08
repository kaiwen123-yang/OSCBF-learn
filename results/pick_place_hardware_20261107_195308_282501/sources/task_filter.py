"""Common velocity CBF/OSCBF filter for the prepared Piper full-task region.

No hardware imports. Both methods share reference, nominal control, inequalities,
quantization, tolerances and rejection rules. Only the QP objective metric differs.
The OSCBF metric is the paper IV-B full-rank square-J special case of Eq. (25).
"""
import time
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import qpax
from piper_oscbf.task_geometry import TaskGeometry


class TaskFilterError(RuntimeError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.reason, self.diagnostics = reason, diagnostics


def quantize_sdk(values):
    """Exactly the existing PiperDevice Python rounding; no later scaling."""
    return np.asarray([round(float(v)*1000)/1000 for v in values], dtype=float)


class TaskSafetyFilter:
    def __init__(self, cfg, method):
        if method not in ('cbf', 'oscbf'):
            raise ValueError('TaskSafetyFilter method must be cbf or oscbf')
        self.cfg, self.method = cfg, method
        opt = cfg.get('safety_filter', {})
        self.alpha_geometry = float(opt.get('alpha_geometry', 2.))
        self.alpha_joint = float(opt.get('alpha_joint', 3.))
        self.kp_joint = np.broadcast_to(np.asarray(opt.get('kp_joint', 2.), float), (6,)).copy()
        self.task_weights = np.asarray(opt.get('task_weights', [1.]*6), float)
        self.objective_scale = float(opt.get('objective_scale', 100.))
        self.quantum = float(opt.get('velocity_quantization_rad_s', .001))
        self.sigma_min_guard = float(opt.get('sigma_min_guard', .10))
        self.solver_tol = float(opt.get('solver_tol', 1e-6))
        self.residual_tol = float(opt.get('residual_tol', 1e-6))
        self.physical_tolerance_m = float(opt.get('physical_tolerance_m', 1e-8))
        self.max_iter = int(opt.get('solver_max_iter', 50))
        positive = [self.alpha_geometry, self.alpha_joint, self.objective_scale,
                    self.sigma_min_guard, self.solver_tol, self.residual_tol]
        if (not np.all(np.isfinite(positive)) or min(positive) <= 0
                or not np.all(np.isfinite(self.kp_joint)) or np.any(self.kp_joint <= 0)
                or self.task_weights.shape != (6,) or not np.all(np.isfinite(self.task_weights))
                or np.any(self.task_weights <= 0) or self.max_iter < 1
                or not np.isfinite(self.physical_tolerance_m) or self.physical_tolerance_m < 0):
            raise ValueError('Positive finite filter settings required')
        if abs(self.quantum-.001) > 1e-15:
            raise ValueError('This fixed SDK uses 0.001 rad/s CPV rounding')
        self.geometry = TaskGeometry(cfg)
        self.geometry_labels = self.geometry.constraint_labels
        self.geometry_constraint_count = len(self.geometry_labels)
        self.constraint_labels = (*self.geometry_labels,
            *(f'joint_lower:joint{j}' for j in range(1,7)),
            *(f'joint_upper:joint{j}' for j in range(1,7)),
            *(f'velocity_positive:joint{j}' for j in range(1,7)),
            *(f'velocity_negative:joint{j}' for j in range(1,7)))
        self._lower = jnp.asarray(self.geometry.tcp_robot.joint_lower_limits)
        self._upper = jnp.asarray(self.geometry.tcp_robot.joint_upper_limits)
        eye, kp = jnp.eye(6), jnp.asarray(self.kp_joint)
        weights = jnp.asarray(self.task_weights)**2

        @jax.jit
        def features(q, q_ref, qd_ref, width, cap, carrying, support_contact,
                     payload_T_tcp, world_cube_T, allow_finger_cube_contact):
            args = (q, width, carrying, support_contact, payload_T_tcp,
                    world_cube_T, allow_finger_cube_contact)
            h_geo = self.geometry.barrier_values(*args)
            physical_geo = self.geometry.barrier_values(*args, clearance_scale=0.)
            dh = self.geometry.barrier_jacobian(*args)
            jac = self.geometry.tcp_jacobian(q)
            tcp, ref_tcp = self.geometry.tcp_transform(q), self.geometry.tcp_transform(q_ref)
            u_nom = qd_ref + kp*(q_ref-q)
            u_nom = u_nom/jnp.maximum(1., jnp.max(jnp.abs(u_nom))/cap)
            metric = jac.T @ (weights[:, None]*jac) if self.method == 'oscbf' else eye
            metric = .5*(metric+metric.T)*self.objective_scale
            joint_h = jnp.concatenate((q-self._lower, self._upper-q))
            h = jnp.concatenate((h_geo, joint_h, jnp.full(12, cap)))
            physical_h = jnp.concatenate((physical_geo, joint_h, jnp.full(12, cap)))
            G = jnp.concatenate((-dh, -eye, eye, eye, -eye))
            b = jnp.concatenate((self.alpha_geometry*h_geo, self.alpha_joint*joint_h,
                                 jnp.full(12, cap)))
            b_tight = b - .5*self.quantum*jnp.sum(jnp.abs(G), axis=1)
            mask = jnp.concatenate((self.geometry.constraint_mask(carrying, allow_finger_cube_contact),
                                    jnp.ones(24, dtype=bool)))
            return {'h':h, 'physical_h':physical_h, 'G':G, 'b_physical':b,
                    'b_robust':b_tight, 'u_nom':u_nom, 'metric':metric,
                    'jacobian':jac, 'singular_values':jnp.linalg.svd(jac, compute_uv=False),
                    'tcp_transform':tcp, 'reference_tcp_transform':ref_tcp,
                    'constraint_mask':mask}

        @jax.jit
        def solve(metric, u_nom, G, b):
            return qpax.solve_qp(metric, -metric@u_nom, jnp.zeros((0,6)), jnp.zeros(0),
                                 G, b, solver_tol=self.solver_tol, max_iter=self.max_iter)

        self._features, self._solve = features, solve
        self._warm = False

    def _inputs(self, q, q_ref, qd_ref, width, cap, payload, world):
        vectors = [np.asarray(v, dtype=float) for v in (q,q_ref,qd_ref)]
        if any(v.shape != (6,) or not np.all(np.isfinite(v)) for v in vectors):
            raise ValueError('Finite six-axis state, reference and reference velocity required')
        if not np.isfinite(width) or not 0 <= width <= .1 or not np.isfinite(cap) or cap <= self.quantum/2:
            raise ValueError('Measured gripper width in [0,0.1] m and positive velocity cap required')
        payload = np.asarray(self.geometry.default_payload_T_tcp if payload is None else payload, dtype=float)
        world = np.asarray(self.geometry.world_cube_transform() if world is None else world, dtype=float)
        if any(t.shape != (4,4) or not np.all(np.isfinite(t)) for t in (payload,world)):
            raise ValueError('Finite explicit 4x4 payload/world transforms required')
        return (*vectors,float(width),float(cap),payload,world)

    def step(self, q, q_ref, qd_ref, gripper_width_m, cap, *, carrying=False,
             support_contact=False, payload_T_tcp=None, world_cube_T=None,
             allow_finger_cube_contact=False):
        if not self._warm:
            raise TaskFilterError('warmup_required_before_step', {'method':self.method})
        started = time.perf_counter()
        q,q_ref,qd_ref,width,cap,payload,world = self._inputs(
            q,q_ref,qd_ref,gripper_width_m,cap,payload_T_tcp,world_cube_T)
        values = self._features(q,q_ref,qd_ref,width,cap,bool(carrying),bool(support_contact),
                                payload,world,bool(allow_finger_cube_contact))
        out = {key:np.asarray(value) for key,value in values.items()}  # synchronizes CPU JAX results
        out.update(method=self.method, u_qp=None, u_sdk=None, qp_margin=None, applied_margin=None,
                   converged=False, iterations=0, qp_solved=False, solver_status='not_solved')
        out['sigma_min'] = float(out['singular_values'][-1])
        out['metric_condition'] = float(np.linalg.cond(out['metric']))
        out['nominal_margin'] = out['b_physical'] - out['G']@out['u_nom']
        rounded_nominal = quantize_sdk(out['u_nom'])
        out['rounded_nominal_margin'] = out['b_physical'] - out['G']@rounded_nominal
        geo_mask = out['constraint_mask'][:self.geometry_constraint_count]
        out['min_h_m'] = float(out['h'][:self.geometry_constraint_count][geo_mask].min())
        out['min_physical_distance_m'] = float(out['physical_h'][:self.geometry_constraint_count][geo_mask].min())
        out['position_error_m'] = out['reference_tcp_transform'][:3,3]-out['tcp_transform'][:3,3]
        out['orientation_error_rad'] = float(np.arccos(np.clip(
            (np.trace(out['reference_tcp_transform'][:3,:3].T@out['tcp_transform'][:3,:3])-1)/2,-1.,1.)))

        def reject(reason):
            out['solver_status'], out['solve_ms'] = reason, (time.perf_counter()-started)*1000
            raise TaskFilterError(reason, out)

        if not all(np.all(np.isfinite(out[key])) for key in
                   ('h','physical_h','G','b_physical','u_nom','metric','jacobian')):
            reject('nonfinite_model_or_constraint')
        if out['sigma_min'] < self.sigma_min_guard:
            reject('outside_audited_full_rank_region')
        if out['min_physical_distance_m'] < -self.physical_tolerance_m:
            reject('physical_geometry_not_clear')
        # Exactly-zero contact commands need no fictitious worst-case rounding
        # error. This identical fast path checks both continuous and actual
        # rounded nominal requests against the original inequalities.
        if min(float(out['nominal_margin'].min()), float(out['rounded_nominal_margin'].min())) >= -self.residual_tol:
            out.update(u_qp=out['u_nom'].copy(), converged=True, qp_solved=False,
                       solver_status='nominal_quantized_feasible')
        else:
            solved = self._solve(out['metric'],out['u_nom'],out['G'],out['b_robust'])
            u,_,_,_,converged,iterations = [np.asarray(v) for v in solved]
            out.update(u_qp=u,converged=bool(converged),iterations=int(iterations),qp_solved=True,
                       solver_status='qp_converged' if converged else 'qp_not_converged')
            out['qp_margin'] = out['b_physical'] - out['G']@u
            out['robust_qp_margin'] = out['b_robust'] - out['G']@u
            if not out['converged'] or not np.all(np.isfinite(u)):
                reject('qp_not_converged_or_nonfinite')
        out['u_sdk'] = quantize_sdk(out['u_qp'])
        out['qp_margin'] = out['b_physical'] - out['G']@out['u_qp']
        out['applied_margin'] = out['b_physical'] - out['G']@out['u_sdk']
        out['robust_qp_margin'] = out['b_robust'] - out['G']@out['u_qp']
        if not np.all(np.isfinite(out['u_sdk'])) or float(out['applied_margin'].min()) < -self.residual_tol:
            reject('rounded_request_constraint_violation')
        out['filter_active'] = bool(np.max(np.abs(out['u_sdk']-rounded_nominal)) > 1e-12)
        out['control_modification_norm'] = float(np.linalg.norm(out['u_sdk']-rounded_nominal))
        out['nominal_constraint_violation'] = float(np.max(-out['nominal_margin']))
        out['max_qp_violation'] = float(max(0.,-out['robust_qp_margin'].min())) if out['qp_solved'] else None
        out['max_applied_violation'] = float(max(0.,-out['applied_margin'].min()))
        out['solve_ms'] = (time.perf_counter()-started)*1000
        return out

    def warmup(self, q, cap=None):
        """Compile features AND solver before any caller starts hardware motion."""
        if jax.default_backend() != 'cpu':
            raise ValueError('This validated task filter requires JAX_PLATFORMS=cpu')
        cap = float(self.cfg['joint_velocity_limit_rad_s'] if cap is None else cap)
        q = np.asarray(q, dtype=float)
        payload,world = self.geometry.default_payload_T_tcp,self.geometry.world_cube_transform()
        started = time.perf_counter()
        values = self._features(q,q,np.zeros(6),float(self.cfg['gripper_open_m']),cap,False,False,
                                np.asarray(payload),np.asarray(world),False)
        values = {key:np.asarray(value) for key,value in values.items()}
        if float(values['singular_values'][-1]) < self.sigma_min_guard:
            raise TaskFilterError('warmup_outside_audited_full_rank_region',values)
        # Compilation probe uses a feasible all-zero-gradient RHS bound and the
        # real shape. It is not a scene-feasibility claim or a hardware command.
        solved=self._solve(values['metric'],np.zeros(6),values['G'],np.full(len(values['G']),1.))
        solved=[np.asarray(v) for v in solved]
        if not bool(solved[-2]):
            raise TaskFilterError('warmup_solver_failed',values)
        # Fixed-q gripper checks call geometry's separate JIT entry points;
        # compile their normal/physical and Jacobian paths before motion too.
        self.geometry.evaluate(q,'pregrasp',float(self.cfg['gripper_open_m']),
            payload_T_tcp=np.asarray(payload),world_cube_T=np.asarray(world),
            carrying=False,support_contact=False)
        self._warm=True
        return {'warmed':True,'method':self.method,'compile_seconds':time.perf_counter()-started,
                'constraint_count':len(self.constraint_labels),'hardware_connected':False,
                'probe_scope':'compile-only feasible numeric RHS, not a scene validation'}

    def metadata(self):
        return {'method':self.method,'frame':'base_link','task_frame':'tcp_link',
            'nominal':'shared qd_ref + kp_joint*(q_ref-q), uniform scaling to cap before filtering',
            'kp_joint':self.kp_joint.tolist(),'task_weights':self.task_weights.tolist(),
            'objective':'I' if self.method=='cbf' else 'J_TCP.T @ diag(task_weights**2) @ J_TCP',
            'objective_scale':self.objective_scale,
            'paper_scope':'OSCBF IV-B Eq.25 full-rank square-J special case; N=0 only inside guarded region',
            'rank_policy':{'sigma_min_guard':self.sigma_min_guard,'regularization':None,
                           'meaning':'engineering guard from prepared-path audit, not manufacturer limit'},
            'alpha_geometry':self.alpha_geometry,'alpha_joint':self.alpha_joint,
            'velocity_quantization_rad_s':self.quantum,
            'quantization_policy':'robust half-quantum L1 erosion for QP; Python SDK rounding; final original-margin recheck; no post-QP scaling',
            'nominal_fast_path':'both continuous and rounded nominal satisfy original constraints; explicitly qp_solved=False',
            'solver_tol':self.solver_tol,'residual_tol':self.residual_tol,
            'physical_tolerance_m':self.physical_tolerance_m,'solver_max_iter':self.max_iter,
            'geometry_constraint_count':self.geometry_constraint_count,
            'constraint_labels':self.constraint_labels,
            'h_row_semantics':'geometry distances [m], joint-limit state margins [rad], then velocity bounds cap [rad/s] (last 12 are not state CBFs)',
            'physical_h_semantics':'same layout, with virtual geometry clearances removed; model-derived conservative distances',
            'timing_scope':'solve_ms includes feature evaluation, host synchronization, solve if required, and final rounding checks',
            'backend':'cpu',
            'warmup_required_before_motion':True}
