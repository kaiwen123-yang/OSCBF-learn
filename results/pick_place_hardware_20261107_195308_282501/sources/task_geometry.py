"""Model-based runtime geometry for the saved Piper pick/place task.

No device or ROS imports. All lengths are metres in base_link unless a frame is
named explicitly. Collision meshes supply conservative local boxes; a union of
cell-circumscribing spheres covers each entire box, not merely its vertices.
The table is conservatively extended to an infinite horizontal support plane.
Self-collision, the fixed base, cables and unmodelled objects are not covered.
"""
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from oscbf.core.manipulator import Manipulator

ROOT = Path(__file__).resolve().parents[1]
FINGERS = frozenset(('gripper_link1', 'gripper_link2'))


def _origin(element):
    out = np.eye(4)
    if element is not None:
        out[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        out[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return out


def box_corners(lower, upper):
    return np.asarray(list(itertools.product(*zip(lower, upper))), dtype=float)


def box_cover(lower, upper, cell_length):
    """Each sphere covers an entire slab of the local mesh bounding box."""
    extent = np.asarray(upper) - np.asarray(lower)
    axis = int(np.argmax(extent))
    count = max(1, int(np.ceil(extent[axis] / cell_length)))
    cell_extent = extent.copy()
    cell_extent[axis] /= count
    centers = np.tile((np.asarray(lower) + upper) / 2, (count, 1))
    centers[:, axis] = lower[axis] + (np.arange(count) + .5) * cell_extent[axis]
    radii = np.full(count, np.linalg.norm(cell_extent / 2))
    return centers, radii, axis, cell_extent


def signed_point_box_distance(points, half_extents):
    """Conservative signed distance to a box at the origin.

    The regularized outside norm is a <= 1e-12 m underestimate and has finite
    derivatives at zero. Interior medial axes and exact edge ties are nonsmooth;
    JAX supplies subgradients there. Negative physical distances are reported,
    not repaired or interpreted as a certified collision-free starting state.
    """
    delta = jnp.abs(points) - jnp.asarray(half_extents)
    outside = jnp.maximum(delta, 0.)
    return (jnp.sqrt(jnp.sum(outside * outside, axis=-1) + 1e-24) - 1e-12
            + jnp.minimum(jnp.max(delta, axis=-1), 0.))


def _points_in_box_frame(points, transform):
    return (points - transform[:3, 3]) @ transform[:3, :3]


def load_task_model(cfg):
    """Use the same nominal flange-to-tip transform as the MoveIt task."""
    metadata = json.loads((ROOT / 'models/piper_h_oscbf_model.json').read_text())
    flange_offset = np.asarray(metadata['ee_offset'], dtype=float)
    xyz = np.asarray(cfg['tcp_offset_m'], dtype=float)
    rpy = np.asarray(cfg.get('tcp_rpy_rad', [0., 0., 0.]), dtype=float)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.all(np.isfinite(np.r_[xyz, rpy])):
        raise ValueError('Finite three-component nominal TCP translation and rotation required')
    tip = np.eye(4)
    tip[:3, 3] = xyz
    tip[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    robot = Manipulator.from_urdf(str(ROOT / 'models/piper_h_oscbf.urdf'),
                                  ee_offset=flange_offset @ tip)
    return robot, {**metadata, 'task_frame': 'tcp_link', 'tcp_offset_m': cfg['tcp_offset_m'],
                   'tcp_status': cfg.get('tcp_status', 'unverified'), 'tcp_calibrated': False,
                   'ee_offset': (flange_offset @ tip).tolist()}


class TaskGeometry:
    """Fixed-shape h(q) and dh/dq, with explicit held/world-object state.

    gripper_width_m is the measured physical opening at runtime, not the saved
    plan's nominal width. The arm Jacobian treats width as fixed in an arm step;
    check_gripper_sweep guards commanded opening changes at a stationary q.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.get('frame', 'base_link') != 'base_link':
            raise ValueError('Runtime geometry requires base_link scene coordinates')
        opts = cfg.get('runtime_geometry', {})
        self.cell_length = float(opts.get('sphere_cell_length_m', .04))
        self.padding = float(opts.get('model_padding_m', 0.))
        if not np.all(np.isfinite([self.cell_length, self.padding])) or self.cell_length <= 0 or self.padding < 0:
            raise ValueError('Positive cell length and nonnegative model padding required')
        self.obstacle_clearance = float(opts.get('obstacle_clearance_m', 0.))
        self.robot_table_clearance = float(opts.get('robot_table_clearance_m', 0.))
        self.payload_table_clearance = float(opts.get('payload_table_clearance_m', 0.))
        self.target_cube_clearance = float(opts.get('target_cube_clearance_m', 0.))
        self.payload_robot_clearance = float(opts.get('payload_robot_clearance_m', 0.))
        gaps = [self.obstacle_clearance, self.robot_table_clearance, self.payload_table_clearance,
                self.target_cube_clearance, self.payload_robot_clearance]
        if not np.all(np.isfinite(gaps)) or min(gaps) < 0:
            raise ValueError('Clearances must not permit penetration')
        self.table_enabled = bool(opts.get('table_enabled', True))
        self.target_cube_enabled = bool(opts.get('target_cube_enabled', True))
        self.payload_robot_enabled = bool(opts.get('payload_robot_enabled', True))
        self.cube_size = float(cfg['cube_size_m'])
        if not np.isfinite(self.cube_size) or self.cube_size <= 0:
            raise ValueError('Cube edge length must be positive')
        if not np.isfinite(cfg['table_top_z_m']) or any(
                np.asarray(cfg[k]).shape != (2,) or not np.all(np.isfinite(cfg[k])) for k in ('pick_xy_m', 'place_xy_m')):
            raise ValueError('Finite table height and two-component pick/place coordinates required')
        self.tcp_robot, self.model_metadata = load_task_model(cfg)
        metadata = json.loads((ROOT / 'models/piper_h_oscbf_model.json').read_text())
        self.flange_robot = Manipulator.from_urdf(str(ROOT / 'models/piper_h_oscbf.urdf'),
                                                 ee_offset=np.asarray(metadata['ee_offset']))
        portable_path = ROOT / 'models/piper_h_fixed_gripper.urdf'
        portable = ET.parse(portable_path).getroot()
        original = ET.parse(ROOT / 'models/piper_h_with_gripper.urdf').getroot()
        joints = {j.get('name'): j for j in original.findall('joint')}
        flange = np.asarray(metadata['ee_offset'])
        base = flange @ _origin(joints['gripper_base_joint'].find('origin'))
        parts, sphere_parts, centers, radii, corners, corner_parts, labels = [], [], [], [], [], [], []
        self.mesh_metadata = []
        self.part_names = []
        for link in portable.findall('link'):
            name = link.get('name')
            if name == 'base_link' or link.find('collision') is None:
                continue
            if name.startswith('link') and name[4:].isdigit():
                parent, relative, width_vector = int(name[4:]) - 1, np.eye(4), np.zeros(3)
            elif name == 'flange_link':
                parent, relative, width_vector = 5, flange, np.zeros(3)
            elif name == 'gripper_base':
                parent, relative, width_vector = 5, base, np.zeros(3)
            elif name in FINGERS:
                joint = joints['gripper_joint' + name[-1]]
                relative = base @ _origin(joint.find('origin'))
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                multiplier = float(joint.find('mimic').get('multiplier'))
                parent, width_vector = 5, relative[:3, :3] @ axis * multiplier
            else:
                raise ValueError(f'Unmapped moving collision link: {name}')
            vertices, mesh_files = [], []
            for collision in link.findall('collision'):
                mesh_tag = collision.find('geometry/mesh')
                if mesh_tag is None:
                    raise ValueError(f'Unsupported collision primitive in {name}')
                path = (portable_path.parent / mesh_tag.get('filename')).resolve()
                mesh = trimesh.load(str(path), force='mesh', process=False)
                scale = np.fromstring(mesh_tag.get('scale', '1 1 1'), sep=' ')
                transform = _origin(collision.find('origin'))
                vertices.append(np.asarray(mesh.vertices) * scale @ transform[:3, :3].T + transform[:3, 3])
                mesh_files.append({'path': str(path.relative_to(ROOT)),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            vertices = np.vstack(vertices)
            lower, upper = vertices.min(axis=0) - self.padding, vertices.max(axis=0) + self.padding
            c, r, axis, cell_extent = box_cover(lower, upper, self.cell_length)
            index = len(parts)
            parts.append((parent, relative, width_vector))
            self.part_names.append(name)
            sphere_parts.extend([index] * len(r)); centers.extend(c); radii.extend(r)
            corners.extend(box_corners(lower, upper)); corner_parts.extend([index] * 8)
            labels.extend(f'{name}/sphere{k}' for k in range(len(r)))
            nearest = np.min(np.linalg.norm(vertices[:, None, :] - c[None, :, :], axis=-1) - r, axis=1)
            self.mesh_metadata.append({'link': name, 'meshes': mesh_files,
                'local_box_lower_m': lower.tolist(), 'local_box_upper_m': upper.tolist(),
                'cover_axis': axis, 'cell_extent_m': cell_extent.tolist(),
                'sphere_centers_m': c.tolist(), 'sphere_radii_m': r.tolist(),
                'max_mesh_vertex_outside_cover_m': float(nearest.max()),
                'coverage_reason': 'Each slab is contained in its circumsphere; slabs tile the full local collision-mesh AABB.'})
        self._parents = jnp.asarray([p[0] for p in parts], dtype=int)
        self._relative = jnp.asarray(np.stack([p[1] for p in parts]))
        self._width_vectors = jnp.asarray(np.stack([p[2] for p in parts]))
        self._centers, self._radii = jnp.asarray(centers), jnp.asarray(radii)
        self._sphere_parts, self._corner_parts = jnp.asarray(sphere_parts), jnp.asarray(corner_parts)
        self._corners = jnp.asarray(corners)
        self.sphere_labels = tuple(labels)
        self._finger_mask = jnp.asarray([self.part_names[p] in FINGERS for p in sphere_parts])
        self._cube_corners = jnp.asarray(box_corners(np.full(3, -self.cube_size/2), np.full(3, self.cube_size/2)))
        self._payload_radius = np.sqrt(3) * self.cube_size / 2
        self.default_payload_T_tcp = np.eye(4)
        self.default_payload_T_tcp[:3, 3] = cfg.get('held_cube_tcp_offset_m',
            [0., 0., -(self.cube_size/2 - float(cfg['tip_clearance_m']))])
        if not np.all(np.isfinite(self.default_payload_T_tcp)):
            raise ValueError('Finite held-cube offset required')
        self.obstacles = []
        for box in cfg.get('obstacles', []):
            center, size = np.asarray(box['center_m'], float), np.asarray(box['size_m'], float)
            if (center.shape != (3,) or size.shape != (3,) or not np.all(np.isfinite(np.r_[center,size]))
                    or np.any(size <= 0) or box.get('frame', 'base_link') != 'base_link'):
                raise ValueError('Obstacle center/size must be finite 3-vectors with positive dimensions')
            box_rpy = np.asarray(box.get('rpy_rad', [0., 0., 0.]), float)
            if box_rpy.shape != (3,) or not np.all(np.isfinite(box_rpy)) or np.any(np.abs(box_rpy) > 1e-12):
                raise ValueError('Static obstacles currently require base_link-aligned boxes')
            self.obstacles.append((str(box['name']), jnp.asarray(center), jnp.asarray(size/2)))
        labels = []
        for name, _, _ in self.obstacles:
            labels.extend(f'obstacle:{name}/{s}' for s in self.sphere_labels)
            labels.append(f'obstacle:{name}/payload')
        labels.extend(f'table:{part}/corner{k}' for part in self.part_names for k in range(8))
        labels.extend(f'table:payload/corner{k}' for k in range(8))
        labels.extend(f'world_cube:{s}' for s in self.sphere_labels)
        labels.extend(f'held_cube_robot:{s}' for s in self.sphere_labels)
        self.constraint_labels = tuple(labels)
        self.barrier_jacobian = jax.jacfwd(self.barrier_values, argnums=0)
        self._compiled_values = jax.jit(self.barrier_values)
        self._compiled_jacobian = jax.jit(self.barrier_jacobian)

    def tcp_transform(self, q):
        return self.tcp_robot.ee_transform(q)

    def tcp_jacobian(self, q):
        return self.tcp_robot.ee_jacobian(q)

    def world_cube_transform(self, at_place=False):
        transform = np.eye(4)
        transform[:3, 3] = [*self.cfg['place_xy_m' if at_place else 'pick_xy_m'],
                             self.cfg['table_top_z_m'] + self.cube_size/2]
        return transform

    def attach_cube(self, q, world_cube_T=None):
        """Freeze relative pose from model FK and supplied scene pose, not metrology."""
        world = jnp.asarray(self.world_cube_transform() if world_cube_T is None else world_cube_T)
        tcp = self.tcp_transform(q)
        inverse = jnp.eye(4).at[:3, :3].set(tcp[:3, :3].T)
        inverse = inverse.at[:3, 3].set(-tcp[:3, :3].T @ tcp[:3, 3])
        return inverse @ world

    def cube_world_transform(self, q, payload_T_tcp=None):
        return self.tcp_transform(q) @ jnp.asarray(
            self.default_payload_T_tcp if payload_T_tcp is None else payload_T_tcp)

    def part_transforms(self, q, gripper_width_m):
        joints = self.flange_robot.joint_to_world_transforms(q)
        relative = self._relative.at[:, :3, 3].add(self._width_vectors * gripper_width_m)
        return joints[self._parents] @ relative

    def world_geometry(self, q, gripper_width_m):
        transforms = self.part_transforms(q, gripper_width_m)
        sphere_tf, corner_tf = transforms[self._sphere_parts], transforms[self._corner_parts]
        sphere_centers = jnp.einsum('nij,nj->ni', sphere_tf[:, :3, :3], self._centers) + sphere_tf[:, :3, 3]
        corners = jnp.einsum('nij,nj->ni', corner_tf[:, :3, :3], self._corners) + corner_tf[:, :3, 3]
        return sphere_centers, corners

    def constraint_mask(self, carrying, allow_finger_cube_contact=False):
        n = len(self.sphere_labels)
        masks = []
        for _ in self.obstacles:
            masks.extend((jnp.ones(n, bool), jnp.asarray(carrying).reshape(1)))
        masks.extend((jnp.full(len(self.part_names)*8, self.table_enabled),
                      jnp.full(8, self.table_enabled) & carrying,
                      jnp.full(n, self.target_cube_enabled) & ~jnp.asarray(carrying)
                      & ~(self._finger_mask & allow_finger_cube_contact),
                      jnp.full(n, self.payload_robot_enabled) & carrying & ~self._finger_mask))
        return jnp.concatenate(masks)

    def width_lipschitz_bounds(self):
        """Distance variation per metre of gripper opening; fixed q and payload pose."""
        finger_bounds = self._finger_mask.astype(float) * .5
        bounds = []
        for _ in self.obstacles:
            bounds.extend((finger_bounds, jnp.zeros(1)))
        corner_bounds = jnp.asarray([.5 if name in FINGERS else 0.
                                     for name in self.part_names for _ in range(8)])
        bounds.extend((corner_bounds, jnp.zeros(8), finger_bounds, finger_bounds))
        return jnp.concatenate(bounds)

    def barrier_values(self, q, gripper_width_m, carrying, support_contact,
                       payload_T_tcp=None, world_cube_T=None,
                       allow_finger_cube_contact=False, clearance_scale=1.0):
        centers, corners = self.world_geometry(q, gripper_width_m)
        payload = self.cube_world_transform(q, payload_T_tcp)
        payload_corners = self._cube_corners @ payload[:3, :3].T + payload[:3, 3]
        world_cube = jnp.asarray(self.world_cube_transform() if world_cube_T is None else world_cube_T)
        values = []
        for _, center, half_size in self.obstacles:
            values.append(signed_point_box_distance(centers - center, half_size) - self._radii
                          - clearance_scale * self.obstacle_clearance)
            values.append((signed_point_box_distance(payload[:3, 3] - center, half_size)
                           - self._payload_radius - clearance_scale*self.obstacle_clearance).reshape(1))
        values.append(corners[:, 2] - self.cfg['table_top_z_m'] - clearance_scale*self.robot_table_clearance)
        cube_table_gap = jnp.where(support_contact, 0., self.payload_table_clearance)
        values.append(payload_corners[:, 2] - self.cfg['table_top_z_m'] - clearance_scale*cube_table_gap)
        values.append(signed_point_box_distance(_points_in_box_frame(centers, world_cube), self.cube_size/2)
                      - self._radii - clearance_scale*self.target_cube_clearance)
        values.append(signed_point_box_distance(_points_in_box_frame(centers, payload), self.cube_size/2)
                      - self._radii - clearance_scale*self.payload_robot_clearance)
        h = jnp.concatenate(values)
        return jnp.where(self.constraint_mask(carrying, allow_finger_cube_contact), h, 1.)

    @staticmethod
    def stage_defaults(stage):
        return {'carrying': stage in ('lift', 'transfer', 'lower', 'release'),
                'support_contact': stage in ('grasp', 'lift', 'lower', 'release'),
                'allow_finger_cube_contact': stage in ('approach', 'grasp', 'release', 'retreat'),
                'at_place': stage in ('release', 'retreat', 'return_initial')}

    def evaluate(self, q, stage, gripper_width_m, payload_T_tcp=None,
                 carrying=None, world_cube_T=None, support_contact=None):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.all(np.isfinite(q)) or not np.isfinite(gripper_width_m) or not 0 <= gripper_width_m <= .1:
            raise ValueError('Finite six-axis state and measured gripper opening in [0,0.1] m required')
        flags = self.stage_defaults(stage)
        if carrying is not None: flags['carrying'] = bool(carrying)
        if support_contact is not None: flags['support_contact'] = bool(support_contact)
        world_cube_T = self.world_cube_transform(flags['at_place']) if world_cube_T is None else world_cube_T
        args = (q, gripper_width_m, flags['carrying'], flags['support_contact'], payload_T_tcp,
                world_cube_T, flags['allow_finger_cube_contact'])
        h = np.asarray(self._compiled_values(*args))
        jac = np.asarray(self._compiled_jacobian(*args))
        mask = np.asarray(self.constraint_mask(flags['carrying'], flags['allow_finger_cube_contact']))
        physical = np.asarray(self._compiled_values(*args, clearance_scale=0.))
        return {'h': h, 'jacobian': jac, 'active_mask': mask, 'physical_distance_m': physical,
                'labels': self.constraint_labels, 'flags': flags,
                'tcp_transform': np.asarray(self.tcp_transform(q)),
                'min_h_m': float(h[mask].min()), 'min_physical_distance_m': float(physical[mask].min())}

    def check_gripper_sweep(self, q, stage, start_width_m, end_width_m, samples=21, **kwargs):
        """Fixed-q opening guard; each finger translates <= |delta_width| / 2.

        Each width-grid interval is covered by a Lipschitz buffer of a quarter
        of its width. This checks arm geometry only at the supplied fixed q.
        """
        if samples < 2: raise ValueError('At least two width samples required')
        widths = np.linspace(start_width_m, end_width_m, samples)
        half_width_step = abs(end_width_m-start_width_m)/(2*(samples-1))
        checks = [self.evaluate(q, stage, float(w), **kwargs) for w in widths]
        lower_bounds = np.min([c['h'] for c in checks], axis=0) - np.asarray(self.width_lipschitz_bounds())*half_width_step
        active = checks[0]['active_mask']
        minimum = min(c['min_h_m'] for c in checks)
        lower_bound = float(lower_bounds[active].min())
        return {'width_start_m': float(start_width_m), 'width_end_m': float(end_width_m),
                'samples': samples, 'maximum_between_sample_lipschitz_buffer_m': half_width_step*.5,
                'minimum_sample_h_m': minimum, 'certified_lower_bound_m': lower_bound,
                'clear_for_fixed_q': lower_bound >= 0.,
                'negative_within_1e_minus8_m_only': -1e-8 <= lower_bound < 0.,
                'coverage': 'Known boxes/table/cube pairs only; measured width and fixed arm state required.'}

    def metadata(self):
        return {'frame': 'base_link', 'task_frame': 'tcp_link', 'tcp_offset_m': self.cfg['tcp_offset_m'],
                'tcp_calibrated': False, 'geometry_source': 'Official fixed-revision collision meshes',
                'moving_links': self.part_names, 'sphere_count': len(self.sphere_labels),
                'constraint_count': len(self.constraint_labels), 'constraint_labels': self.constraint_labels,
                'settings': {'sphere_cell_length_m': self.cell_length, 'model_padding_m': self.padding,
                             'obstacle_clearance_m': self.obstacle_clearance,
                             'robot_table_clearance_m': self.robot_table_clearance,
                             'payload_table_clearance_m': self.payload_table_clearance,
                             'target_cube_clearance_m': self.target_cube_clearance,
                             'payload_robot_clearance_m': self.payload_robot_clearance},
                'default_payload_T_tcp': self.default_payload_T_tcp.tolist(),
                'enabled_pairs': {'moving_robot_to_static_boxes': True, 'moving_robot_to_table': self.table_enabled,
                                  'held_cube_to_static_boxes': True, 'held_cube_to_table': self.table_enabled,
                                  'moving_robot_to_world_cube': self.target_cube_enabled,
                                  'held_cube_to_nonfinger_robot': self.payload_robot_enabled},
                'mesh_coverage': self.mesh_metadata,
                'table_model': 'Infinite horizontal plane, conservative beyond finite table footprint',
                'payload_assumption': 'Rigid cube attachment; model FK/scene-derived pose, no slip measurement',
                'gripper_width': 'Actual SDK feedback required at runtime; plan width only for offline checks',
                'arm_qp_assumption': 'Gripper width is settled during arm velocity steps; width motion is separately checked at fixed q before gripper actuation.',
                'permitted_contact': 'Finger/target-cube during approach/grasp/release/retreat; finger/held-cube while carrying; cube/table extra clearance zero in support-contact stages, nonpenetration retained',
                'omitted_pairs': ['robot self-collision', 'fixed base', 'cables and unmodelled environment'],
                'distance_status': 'Conservative model calculations, not external metrology'}
