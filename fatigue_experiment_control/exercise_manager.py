import os
import copy
import pickle
import threading

from rclpy.node import Node
from dataclasses import dataclass, asdict
from enum import IntEnum, auto
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration

from ur10e_custom_control.ur10e_configs import UR_QOS_PROFILE

@dataclass
class Exercise:
    name: str
    poses: list[PoseStamped]
    joint_angles: list[list[float]]
    duration: list[Duration]

class ExerciseType(IntEnum):
    HORIZONTAL_BICEP_CURL = auto()
    VERTICAL_BICEP_CURL = auto()
    
    HORIZONTAL_LATERAL_RAISE = auto()
    VERTICAL_LATERAL_RAISE = auto()

class ExerciseManager:
    def __init__(self, node: Node, distance_threshold: float):
        self._node = node
        self._active_exercise: Exercise = None
        self._distance_threshold = distance_threshold

        self._pose_decimation = 1
        self._pose_counter = 0

        self._joint_decimation = 1
        self._joint_counter = 0
        
        self._monitor_request = False

        self._joint_angle_sub = self._node.create_subscription(
            msg_type=JointState, 
            topic="/joint_states",
            callback=self.add_joint_state,
            qos_profile=UR_QOS_PROFILE
        )

        self._pose_sub = self._node.create_subscription(
            msg_type=PoseStamped,
            topic="tcp_pose_broadcaster/pose",
            callback=self.add_pose,
            qos_profile=UR_QOS_PROFILE
        )

        self._lock = threading.Lock()

    def get_exercise(self, decimated: bool = True) -> Exercise | None:
        if self._active_exercise is None or self._monitor_request: 
            return None
        
        exercise = copy.deepcopy(self._active_exercise)
        if decimated:
            exercise.poses = self.generate_path(exercise.poses)

        return exercise
    
    def save_exercise(self, fp: os.PathLike) -> bool:
        if self._active_exercise is None or self._monitor_request:
            return False
        
        self._active_exercise.name = os.path.basename(fp).split(".")[0]
        with open(fp, "wb") as _pickled_file:
            pickle.dump(asdict(self._active_exercise), _pickled_file)

        return True

    def load_exercise(self, fp: os.PathLike):
        try:
            self._active_exercise = pickle.load(fp)
        except Exception:
            return False

    def start_monitoring(self, name: str, pose_decimation: int = 1, joint_decimation: int = 1):
        with self._lock:
            self._active_exercise = Exercise(
                name=name,
                poses=[],
                joint_angles=[],
                duration=[]
            )

            self._pose_decimation = pose_decimation
            self._pose_counter = 0

            self._joint_decimation = joint_decimation
            self._joint_counter = 0

            self._monitor_request = True

    def stop_monitoring(self):
        with self._lock:
            self._monitor_request = False

    def add_pose(self, pose: PoseStamped):
        with self._lock:
            if self._active_exercise is not None and self._monitor_request:
                self._pose_counter += 1
                if self._pose_counter % self._pose_decimation == 0:
                    self._active_exercise.poses.append(pose)

    def add_joint_state(self, joints: JointState):
        with self._lock:
            if self._active_exercise is not None and self._monitor_request:
                self._joint_counter += 1
                if self._joint_counter % self._joint_decimation == 0:
                    self._active_exercise.joint_angles.append(joints.position)

    def generate_path(self, poses: list[PoseStamped]) -> list[PoseStamped]:
        def __dist3d(_t1: PoseStamped, _t2: PoseStamped) -> float:
            return ((_t1.pose.position.x-_t2.pose.position.x)**2 + \
                    (_t1.pose.position.y-_t2.pose.position.y)**2 + \
                    (_t1.pose.position.z-_t2.pose.position.z)**2)**0.5
        
        last_pose: PoseStamped = poses[0]
        trajectory: list[PoseStamped] = [] # Do not include initial pose in the trajectory
        for pose in poses:
            if __dist3d(last_pose, pose) >= self._distance_threshold:
                last_pose = pose
                trajectory.append(pose)

        return trajectory
    
    def get_exercise_dofs(self, exercise: ExerciseType):
        match exercise:
            case ExerciseType.HORIZONTAL_BICEP_CURL:
                # Only enable translation in the xy-plane
                return [True, True, False, False, False, False]
            case ExerciseType.VERTICAL_BICEP_CURL:
                # Only enable translation in yz-plane
                return [False, True, True, False, False, False]
            case ExerciseType.HORIZONTAL_LATERAL_RAISE:
                # Only enable translation in the xy-plane and rotation in the y-axis
                return [True, True, False, False, True, False]
            case ExerciseType.VERTICAL_LATERAL_RAISE:
                # Only enable translation in the yz-plane and rotation in the y-axis
                return [False, True, True, False, True, False]
            case _:
                raise ValueError(f"Unsupported exercise type: {exercise}")