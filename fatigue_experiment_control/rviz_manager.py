import numpy as np
import threading
import transforms3d

from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Point, Wrench, Twist
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
from std_msgs.msg import ColorRGBA

from fatigue_experiment_control.exercise_manager import Exercise

from PyQt5.QtCore import Qt, QMetaObject

def pose_to_ndarray(pose: Pose):
    rotmat = transforms3d.quaternions.quat2mat([
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z
    ])

    return np.array([
        [rotmat[0,0], rotmat[0,1], rotmat[0,2], pose.position.x],
        [rotmat[1,0], rotmat[1,1], rotmat[1,2], pose.position.y],
        [rotmat[2,0], rotmat[2,1], rotmat[2,2], pose.position.z],
        [0, 0, 0, 1]
    ])

from PyQt5.QtWidgets import QProgressDialog

class ExerciseProgressWidget(QProgressDialog):
    def __init__(self):
        super().__init__("Processing...", "Cancel", 0, 100)

class RvizManager:
    def __init__(self, 
                 node: Node,
                 refresh_rate: float | None, 
                 move_camera_with_exercise: bool, 
                 align_with_path: bool):
        
        self._node = node
        self._pose: PoseStamped = None
        self._refresh_rate = refresh_rate
        self._move_camera_with_exercise = move_camera_with_exercise
        self._align_with_path = align_with_path

        # self._target_pose_publisher = self._node.create_publisher(PoseStamped, "dynamic_force_target_pose", 0)
        self._target_pose_publisher = self._node.create_publisher(Marker, "dynamic_force_target_pose_marker", 0)
        self._path_publisher = self._node.create_publisher(Path, "dynamic_force_path", 0)
        self._error_vector_publisher = self._node.create_publisher(Marker, "dynamic_force_error_vector", 0)
        self._progress_text_publisher = self._node.create_publisher(Marker, "dynamic_force_progress_text", 0)
        self._rviz_camera_tform_publisher = TransformBroadcaster(self._node, 0)

        self._right_handed = True
        # self._progress_widget = ExerciseProgressWidget()
        # self._progress_widget.show()

        self._lock = threading.Lock()

        if self._refresh_rate is not None:
            assert not align_with_path, "Refresh rate should not be specified if aligning with the path"
            self._camera_follower = self._node.create_timer(
                timer_period_sec=self._refresh_rate,
                callback=self._compute_and_send_camera_frame,
            )

    def set_handedness(self, right_handed: bool):
        with self._lock:
            self._right_handed = right_handed

    @property
    def _constant_transform(self):
        if self._right_handed:
            return pose_to_ndarray(
                Pose(
                    position=Point(x=0.0, y=0.0, z=0.0),
                    orientation=Quaternion(w=0.0, x=0.0, y=1.0, z=0.0)
                )
            )
        else:
            pose_to_ndarray(
                Pose(
                    position=Point(x=0.0, y=0.0, z=0.0),
                    orientation=Quaternion(w=0.0, x=0.0, y=1.0, z=0.0)
                )
            )

    def update_pose(self, pose: PoseStamped):
        with self._lock:
            self._pose = pose

    def _compute_and_send_camera_frame(self, frame: TransformStamped | Pose | PoseStamped | None = None):
        if frame is None:
            with self._lock:
                if self._pose is None: return
                frame = self._pose


        if self._align_with_path:
            if isinstance(frame, TransformStamped): 
                transform = frame
            elif isinstance(frame, Pose): 
                transform = Transform(
                    translation = Vector3(
                        x=frame.position.x,
                        y=frame.position.y,
                        z=frame.position.z
                    ),
                    rotation = frame.orientation
                )
            elif isinstance(frame, PoseStamped):
                transform = Transform(
                    translation = Vector3(
                        x=frame.pose.position.x,
                        y=frame.pose.position.y,
                        z=frame.pose.position.z
                    ),
                    rotation = frame.pose.orientation
                )
            else:
                raise ValueError
        else:
            if isinstance(frame, TransformStamped): 
                raise ValueError
            elif isinstance(frame, Pose): 
                nd_tform = pose_to_ndarray(frame) @ self._constant_transform
            elif isinstance(frame, PoseStamped):
                nd_tform = pose_to_ndarray(frame.pose) @ self._constant_transform
            else:
                raise ValueError
            
            q_new = transforms3d.quaternions.mat2quat(nd_tform[:3,:3])
            
            transform = Transform(
                translation = Vector3(
                    x=nd_tform[0,3],
                    y=nd_tform[1,3],
                    z=nd_tform[2,3],
                ),
                rotation = Quaternion(w=q_new[0], x=q_new[1], y=q_new[2], z=q_new[3])
            )
            
            self._rviz_camera_tform_publisher.sendTransform(
                    TransformStamped(
                    header=Header(frame_id="base"),
                    child_frame_id="tf_dynamic_path",
                    transform=transform
                )
            )

    from ur_msgs.action._dynamic_force_mode_path import DynamicForceModePath_FeedbackMessage
    def exercise_feedback(self, feedback: DynamicForceModePath_FeedbackMessage):
        if self._move_camera_with_exercise:
            if self._align_with_path:
                t1 = feedback.feedback.pose_actual
                t2 = feedback.feedback.pose_desired
                
                y = np.array([0.0, 1.0, 0.0])

                x = np.array([
                    t1.position.x-t2.position.x,
                    t1.position.y-t2.position.y,
                    t1.position.z-t2.position.z
                ])

                # Align the frames along the x-axis, since this is the axis tracked by rviz's third-person follower
                x /= np.linalg.norm(x)
                
                y -= (np.dot(x, y)*x)
                y /= np.linalg.norm(y)
                z = np.cross(x, y)

                q_new = transforms3d.quaternions.mat2quat(
                    np.array([x, y, z]).T
                )

                self._compute_and_send_camera_frame(Transform(
                    translation = Vector3(
                        x=feedback.feedback.pose_actual.position.x,
                        y=feedback.feedback.pose_actual.position.y,
                        z=feedback.feedback.pose_actual.position.z
                    ),
                    rotation = Quaternion(w=q_new[0], x=q_new[1], y=q_new[2], z=q_new[3])
                ))
            elif self._refresh_rate is None:
                self._compute_and_send_camera_frame(feedback.feedback.pose_actual)
        
        # TODO(george): this should probably just already be a PoseStamped object
        target_pose = PoseStamped(
            header=Header(frame_id="base"),
            pose=feedback.feedback.pose_desired,
        )

        error_vector = Marker(
            header=Header(frame_id='base'),
            type=Marker.ARROW,
            action=Marker.ADD,
            scale=Vector3(x=0.01, y=0.02, z=0.01),
            points=[
                Point(x=feedback.feedback.pose_actual.position.x,
                        y=feedback.feedback.pose_actual.position.y,
                        z=feedback.feedback.pose_actual.position.z),
                Point(x=feedback.feedback.pose_desired.position.x,
                        y=feedback.feedback.pose_desired.position.y,
                        z=feedback.feedback.pose_desired.position.z)
            ],
            lifetime=Duration(sec=1),
            color=ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0)
        )

        # self._target_pose_publisher.publish(target_pose)
        self._target_pose_publisher.publish(
            Marker(
                header=Header(frame_id='base'),
                type=Marker.LINE_LIST,
                action=Marker.ADD,
                scale=Vector3(x=0.01, y=0.00, z=0.00),
                points=[
                    Point(),
                    Point(x=0.1, 
                          y=0.0, 
                          z=0.0),
                    Point(),
                    Point(x=0.0, 
                          y=0.1, 
                          z=0.0),
                    Point(),
                    Point(x=0.0, 
                          y=0.0,
                          z=0.1),
                ],
                pose=feedback.feedback.pose_desired,
                lifetime=Duration(sec=1),
                colors=[
                    ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0),
                    ColorRGBA(r=1.0,g=0.0,b=0.0,a=1.0),
                    ColorRGBA(r=0.0,g=1.0,b=0.0,a=1.0),
                    ColorRGBA(r=0.0,g=1.0,b=0.0,a=1.0),
                    ColorRGBA(r=0.0,g=0.0,b=1.0,a=1.0),
                    ColorRGBA(r=0.0,g=0.0,b=1.0,a=1.0)
                ]
            )
        )

        self._error_vector_publisher.publish(error_vector)
        # self._progress_widget.setValue(int(feedback.feedback.progress_percentage * 100))

        self._progress_text_publisher.publish(
            Marker(
                header=Header(frame_id='wrist_3_link'),
                lifetime=Duration(sec=1),
                type=Marker.TEXT_VIEW_FACING,
                pose=Pose(position=Point(z=-0.25)),
                scale=Vector3(x=0.1, y=0.0, z=0.1),
                text=f"Progress [{'active' if not feedback.feedback.is_paused else 'paused'}]: {(feedback.feedback.progress_percentage * 100):.2f}%",
                color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            )
        )

    def reset(self):
        # if not self._progress_widget.isHidden():
        #     QMetaObject.invokeMethod(self._progress_widget, 'hide', Qt.ConnectionType.QueuedConnection)

        self._progress_text_publisher.publish(
            Marker(
                header=Header(frame_id='wrist_3_link'),
                lifetime=Duration(sec=1),
                type=Marker.TEXT_VIEW_FACING,
                pose=Pose(position=Point(z=-0.25)),
                scale=Vector3(x=0.0, y=0.0, z=0.1),
                text="Paused",
                color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            )
        )

    def visualize_exercise(self, exercise: Exercise):
        self._path_publisher.publish(Path(header=Header(frame_id="base"), poses=exercise.poses))
