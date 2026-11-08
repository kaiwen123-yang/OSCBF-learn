"""Plan a complete fixed pick/place task with MoveIt; never connect to CAN.

Uses the existing Piper H model including the movable gripper. Cartesian
approach/lift/descent/retreat and collision-checked OMPL transfers are saved
together with world/attached-object transitions. An example is NOT a real scene.
"""
import argparse
import copy
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ["ROS_DOMAIN_ID"] = "78"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject, CollisionObject,
                             Constraints, JointConstraint, PlanningScene, PlanningSceneComponents, RobotState)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetMotionPlan,
                            GetPlanningScene, GetPositionIK, GetPositionFK, GetStateValidity)
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGERS = ["gripper_link1", "gripper_link2"]


def pose(xyz, downward=False):
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, xyz)
    if downward:
        p.orientation.y, p.orientation.w = 1.0, 0.0
    else:
        p.orientation.w = 1.0
    return p


def box(name, xyz, size, frame="base_link"):
    obj = CollisionObject()
    obj.id, obj.header.frame_id, obj.operation = name, frame, CollisionObject.ADD
    shape = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(map(float, size)))
    obj.primitives, obj.primitive_poses = [shape], [pose(xyz)]
    return obj


class Planner:
    def __init__(self, cfg):
        self.cfg, self.q, self.attached = cfg, list(map(float, cfg["initial_q_rad"])), None
        self.width = cfg["gripper_open_m"]
        self.node = rclpy.create_node("piper_pick_place_planner", namespace="piper_task")
        self.clients = {}
        for name, typ in [("apply_planning_scene", ApplyPlanningScene),
                          ("compute_ik", GetPositionIK), ("plan_kinematic_path", GetMotionPlan),
                          ("compute_fk", GetPositionFK),
                          ("compute_cartesian_path", GetCartesianPath),
                          ("get_planning_scene", GetPlanningScene),
                          ("check_state_validity", GetStateValidity)]:
            client = self.node.create_client(typ, name)
            if not client.wait_for_service(timeout_sec=15):
                raise RuntimeError(f"{name} unavailable; run bash scripts/run_task_planner.sh")
            self.clients[name] = client
        self.steps = []

    def tcp_probe(self):
        request = GetPositionFK.Request()
        request.header.frame_id = 'base_link'
        request.fk_link_names = ['tcp_link']
        request.robot_state = self.state()
        result = self.call('compute_fk', request)
        return {'joint_rad': list(self.q), 'gripper_width_m': self.width,
                'pose': message_to_ordereddict(result.pose_stamped[0]),
                'declared_tcp_offset_m': self.cfg['tcp_offset_m']}

    def call(self, name, request):
        future = self.clients[name].call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=35)
        if not future.done():
            raise RuntimeError(f"{name} timed out")
        return future.result()

    def state(self):
        state = RobotState()
        state.joint_state = JointState(name=JOINTS+["gripper"], position=self.q+[float(self.width)])
        # Explicit state and attachment: never substitute the mock controller state.
        state.is_diff = False
        if self.attached:
            state.attached_collision_objects = [copy.deepcopy(self.attached)]
        return state

    def apply(self, world=(), attached=()):
        scene = PlanningScene(is_diff=True)
        scene.robot_state = self.state()
        scene.robot_state.is_diff = True
        scene.world.collision_objects = list(world)
        scene.robot_state.attached_collision_objects = list(attached)
        response = self.call("apply_planning_scene", ApplyPlanningScene.Request(scene=scene))
        if not response.success:
            raise RuntimeError("Planning scene update failed")

    def initial_scene(self):
        c, size = self.cfg, self.cfg["cube_size_m"]
        table = box("table", [*c["table_center_xy_m"], c["table_top_z_m"]-.025],
                    [*c["table_size_xy_m"], .05])
        cube = box("cube", [*c["pick_xy_m"], c["table_top_z_m"]+size/2], [size]*3)
        obstacles = [box(o["name"], o["center_m"], o["size_m"]) for o in c["obstacles"]]
        removed = AttachedCollisionObject(link_name="tcp_link")
        removed.object.id, removed.object.operation = "cube", CollisionObject.REMOVE
        self.apply([table, cube, *obstacles], [removed])
        self.steps.append({"kind": "scene", "event": "world_initialized",
                           "objects": [message_to_ordereddict(o) for o in [table,cube,*obstacles]]})

    def valid(self):
        req = GetStateValidity.Request(robot_state=self.state(), group_name="arm")
        value = self.call("check_state_validity", req)
        if not value.valid:
            contacts = [(p.contact_body_1, p.contact_body_2) for p in value.contacts]
            raise RuntimeError(f"Invalid start state: {contacts}")

    def support_contact(self, allowed):
        req = GetPlanningScene.Request(components=PlanningSceneComponents(
            components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX))
        matrix = self.call("get_planning_scene", req).scene.allowed_collision_matrix
        for name in ("cube", "table"):
            if name not in matrix.entry_names:
                for entry in matrix.entry_values:
                    entry.enabled.append(False)
                matrix.entry_names.append(name)
                matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(matrix.entry_names)))
        i,j = matrix.entry_names.index("cube"),matrix.entry_names.index("table")
        matrix.entry_values[i].enabled[j] = matrix.entry_values[j].enabled[i] = allowed
        scene = PlanningScene(is_diff=True, allowed_collision_matrix=matrix)
        if not self.call("apply_planning_scene", ApplyPlanningScene.Request(scene=scene)).success:
            raise RuntimeError("Could not update cube/table contact")

    def arm(self, name, target=None, joints=None, cartesian=False):
        self.valid()
        start = self.state()
        if cartesian:
            req = GetCartesianPath.Request()
            req.header.frame_id, req.group_name, req.link_name = "base_link", "arm", "tcp_link"
            req.start_state, req.waypoints = start, [target]
            req.max_step, req.jump_threshold, req.avoid_collisions = .005, 1.5, True
            response = self.call("compute_cartesian_path", req)
            if response.error_code.val != 1 or response.fraction < .999999:
                raise RuntimeError(f"{name}: Cartesian path incomplete fraction={response.fraction}, code={response.error_code.val}")
            trajectory = response.solution
        else:
            if target is not None:
                req = GetPositionIK.Request()
                ik = req.ik_request
                ik.group_name, ik.ik_link_name = "arm", "tcp_link"
                ik.robot_state, ik.avoid_collisions = copy.deepcopy(start), True
                ik.pose_stamped = PoseStamped(pose=target)
                ik.pose_stamped.header.frame_id = "base_link"
                ik.timeout.sec = 3
                response = self.call("compute_ik", req)
                if response.error_code.val != 1:
                    # KDL can get stuck at the singular all-zero seed. This seed
                    # is used only by IK; it is never sent as a motion waypoint.
                    yaw = math.atan2(target.position.y, target.position.x)
                    wrist = yaw
                    ik.robot_state.joint_state.position = [yaw, math.pi/2, -math.pi/2,
                                                           0., math.radians(80), wrist, self.width]
                    response = self.call("compute_ik", req)
                if response.error_code.val != 1:
                    raise RuntimeError(f"{name}: collision-free IK failed, code={response.error_code.val}")
                values = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
                joints = [values[j] for j in JOINTS]
            req = GetMotionPlan.Request()
            m = req.motion_plan_request
            m.group_name, m.pipeline_id = "arm", "ompl"
            m.start_state, m.allowed_planning_time, m.num_planning_attempts = start, 5.0, 5
            m.max_velocity_scaling_factor, m.max_acceleration_scaling_factor = .1, .1
            m.goal_constraints = [Constraints(joint_constraints=[
                JointConstraint(joint_name=j, position=float(v), tolerance_above=.0001,
                                tolerance_below=.0001, weight=1.) for j,v in zip(JOINTS,joints)])]
            response = self.call("plan_kinematic_path", req).motion_plan_response
            if response.error_code.val != 1:
                raise RuntimeError(f"{name}: motion plan failed, code={response.error_code.val}")
            trajectory = response.trajectory
        tr = trajectory.joint_trajectory
        if not tr.points or set(tr.joint_names) != set(JOINTS):
            raise RuntimeError(f"{name}: missing or unexpected joint trajectory")
        end = dict(zip(tr.joint_names, tr.points[-1].positions))
        self.q = [end[j] for j in JOINTS]
        self.valid()
        self.steps.append({"kind": "arm", "name": name,
                           "planner": "cartesian" if cartesian else "ompl",
                           "carrying_cube": self.attached is not None,
                           "start_state": message_to_ordereddict(start),
                           "trajectory": message_to_ordereddict(trajectory)})
        print(name, "points=", len(tr.points), "carrying=", self.attached is not None, flush=True)

    def grip(self, close):
        c, size = self.cfg, self.cfg["cube_size_m"]
        if close:
            # Downward TCP +Z is base -Z. Cube center sits above the finger tips.
            offset = -(size/2-c["tip_clearance_m"])
            self.attached = AttachedCollisionObject(link_name="tcp_link", touch_links=FINGERS)
            self.attached.object = box("cube", [0,0,offset], [size]*3, "tcp_link")
            self.width = size
            # Attaching removes the world instance inside MoveIt. A second
            # REMOVE in the same diff fails because it has already been removed.
            self.apply(attached=[self.attached])
        else:
            removed = AttachedCollisionObject(link_name="tcp_link")
            removed.object.id, removed.object.operation = "cube", CollisionObject.REMOVE
            self.attached, self.width = None, c["gripper_open_m"]
            cube = box("cube", [*c["place_xy_m"], c["table_top_z_m"]+size/2], [size]*3)
            self.apply([cube], [removed])
        self.steps.append({"kind": "gripper", "name": "grasp" if close else "release",
                           "width_m": c["gripper_close_command_m"] if close else c["gripper_open_m"],
                           "force_n": c["gripper_force_n"],
                           "scene_event": "attach_cube" if close else "detach_cube_at_place"})

    def transfer(self, final_target):
        via = self.cfg.get('transfer_via_tcp_m', [])
        if not via:
            self.arm('transfer', final_target)
            return
        first = len(self.steps)
        for index, xyz in enumerate(via):
            self.arm(f'transfer_part_{index}', pose(xyz, True))
        self.arm(f'transfer_part_{len(via)}', final_target)
        parts = self.steps[first:]
        combined = copy.deepcopy(parts[0])
        combined['name'], combined['planner'] = 'transfer', 'ompl_via_concatenated'
        trajectory = combined['trajectory']['joint_trajectory']
        trajectory['points'] = []
        components, offset_ns = [], 0
        for part in parts:
            source = part['trajectory']['joint_trajectory']
            points = copy.deepcopy(source['points'])
            order=[source['joint_names'].index(name) for name in trajectory['joint_names']]
            for point in points:
                for field in ('positions','velocities','accelerations','effort'):
                    if point.get(field):point[field]=[point[field][i] for i in order]
            begin = len(trajectory['points'])
            for point in points:
                t = point['time_from_start']
                ns = offset_ns + int(t['sec'])*1_000_000_000 + int(t['nanosec'])
                point['time_from_start'] = {'sec':ns//1_000_000_000, 'nanosec':ns%1_000_000_000}
                trajectory['points'].append(point)
            last = trajectory['points'][-1]['time_from_start']
            offset_ns = int(last['sec'])*1_000_000_000 + int(last['nanosec']) + 10_000_000
            components.append({'name':part['name'],'first_point':begin,'last_point':len(trajectory['points'])-1})
        combined['interior_via_tcp_m'] = via
        combined['subplans'] = components
        combined['via_semantics'] = 'Interior references in one stage; no required arrival or gripper action at the via. All saved subplan points retained.'
        self.steps[first:] = [combined]

    def run(self):
        c = self.cfg
        self.initial_scene()
        self.support_contact(False)
        self.steps.append({"kind":"gripper", "name":"open", "width_m":c["gripper_open_m"],
                           "force_n":c["gripper_force_n"]})
        z = c["table_top_z_m"]+c["tip_clearance_m"]
        self.arm("pregrasp", pose([*c["pick_xy_m"],z+c["lift_m"]], True))
        if "alignment_tip_height_m" in c:
            self.arm("align", pose([*c["pick_xy_m"],c["table_top_z_m"]+c["alignment_tip_height_m"]], True), cartesian=True)
        self.arm("approach", pose([*c["pick_xy_m"],z], True), cartesian=True)
        self.grip(True)
        self.support_contact(True)  # object starts in intended contact with its support
        self.arm("lift", pose([*c["pick_xy_m"],z+c["lift_m"]], True), cartesian=True)
        self.support_contact(False)
        self.transfer(pose([*c["place_xy_m"],z+c["lift_m"]], True))
        self.support_contact(True)  # allow only cube/table contact on vertical placement
        self.arm("lower", pose([*c["place_xy_m"],z], True), cartesian=True)
        self.grip(False)
        self.support_contact(False)
        self.arm("retreat", pose([*c["place_xy_m"],z+c["lift_m"]], True), cartesian=True)
        self.arm("return_initial", joints=list(map(float,c["initial_q_rad"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/pick_place.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT/"results"/("pick_place_plan_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    report = {"status":"failed", "data_source":"MoveIt offline planning, no hardware motion",
              "scene_status":cfg["scene_status"], "physical_grasp_success":"not_tested", "error":None}
    rclpy.init()
    planner = None
    try:
        planner = Planner(cfg)
        report['planning_tcp_probe'] = planner.tcp_probe()
        planner.run()
        report.update(status="planned", arm_segments=sum(s["kind"]=="arm" for s in planner.steps))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
    finally:
        (out/"task.json").write_text(json.dumps({"config":cfg,"report":report,
                                    "steps":planner.steps if planner else []},indent=2)+"\n")
        (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        if planner:
            planner.node.destroy_node()
        rclpy.shutdown()
        print("RESULT_DIR:",out,flush=True)
    return int(report["status"] != "planned")


if __name__ == "__main__":
    raise SystemExit(main())
