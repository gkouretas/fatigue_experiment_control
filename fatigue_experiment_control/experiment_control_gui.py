import threading
import sys
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from ur10e_custom_control.ur_control_qt import URControlQtWindow
from fatigue_experiment_control.robot_control_manager import RobotControlArbiter, RobotControlStatus
from fatigue_experiment_control.exercise_manager import ExerciseManager, ExerciseType
from fatigue_experiment_control.rviz_manager import RvizManager
from ros2_mindrove.mindrove_configs import MINDROVE_ROS_TOPIC_NAME
from ros2_plux_biosignals.plux_configs import PLUX_ROS_TOPIC_NAME
from pi_user_input_node.user_input_node_config import USER_INPUT_TOPIC_NAME
from idl_definitions.msg import (
    MindroveArmBandEightChannelMsg,
    PluxMsg,
    UserInputMsg
)

from python_utils.ros2_utils.visualization.qt_integration import RclpySpinner
from python_utils.ros2_utils.comms.node_manager import get_realtime_qos_profile
from python_utils.utils.datetime_utils import postfix_string_with_current_time
from python_utils.ros2_utils.comms.rosbag_manager import RosbagManager 

from ur_msgs.srv import SetFreedriveParams

from PyQt5.QtWidgets import *

from functools import partial

_DISTANCE_THRESHOLD = 30/1000
_LHS_JOINT_ANGLES = np.radians([
    -58.93,
    118.38,
    -149.50,
    90.26,
    -90.00,
    263.77
]).tolist()
_RHS_JOINT_ANGLES = np.radians([    
    -39.06,
    104.72,
    -155.72,
    90.21,
    -90.10,
    267.40
]).tolist()
_USE_TOOL = True
_NUM_REPETITIONS = 11

class ManualExperimentControlGui(QMainWindow):
    def __init__(self, node: Node):
        super().__init__()

        self._node = node

        self._input_device_watchdog = QLabel(text="Input device: inactive")
        self._mindrove_watchdog = QLabel(text="Mindrove: inactive")
        self._plux_watchdog = QLabel(text="Plux: inactive")

        self._mindrove_sub = self._node.create_subscription(
            msg_type=MindroveArmBandEightChannelMsg,
            topic=MINDROVE_ROS_TOPIC_NAME,
            callback=partial(self._update_watchdog, self._mindrove_watchdog),
            qos_profile=get_realtime_qos_profile(),
            callback_group=ReentrantCallbackGroup()
        )

        self._plux_sub = self._node.create_subscription(
            msg_type=PluxMsg,
            topic=PLUX_ROS_TOPIC_NAME,
            callback=partial(self._update_watchdog, self._plux_watchdog),
            qos_profile=get_realtime_qos_profile(),
            callback_group=ReentrantCallbackGroup()
        )

        self._input_device_sub = self._node.create_subscription(
            msg_type=UserInputMsg,
            topic=USER_INPUT_TOPIC_NAME,
            callback=partial(self._update_watchdog, self._input_device_watchdog),
            qos_profile=get_realtime_qos_profile(),
            callback_group=ReentrantCallbackGroup()
        )

        self._mindrove_watchdog_timer = self._node.create_timer(
            1.0,
            callback=partial(self._update_watchdog, self._mindrove_watchdog, False),
            callback_group=ReentrantCallbackGroup()
        )

        self._plux_watchdog_timer = self._node.create_timer(
            1.0,
            callback=partial(self._update_watchdog, self._plux_watchdog, False),
            callback_group=ReentrantCallbackGroup()
        )

        self._input_device_watchdog_timer = self._node.create_timer(
            1.0,
            callback=partial(self._update_watchdog, self._input_device_watchdog, False),
            callback_group=ReentrantCallbackGroup()
        )

        self._logger_timer = self._node.create_timer(
            0.1,
            callback=self._check_statuses,
            callback_group=ReentrantCallbackGroup()
        )

        self._log_manager = RosbagManager(
            node=self._node
        )

        exercise_tab_layout = QVBoxLayout()

        labels = (
            self._input_device_watchdog,
            self._mindrove_watchdog,
            self._plux_watchdog
        )
        
        for label in labels:
            exercise_tab_layout.addWidget(label)

        self._status = [False, False, False]

        widget = QWidget()
        widget.setLayout(exercise_tab_layout)

        self.setCentralWidget(widget)

    def _update_watchdog(self, label: QLabel, status: bool | PluxMsg | MindroveArmBandEightChannelMsg | UserInputMsg):
        if "Input" in label.text(): 
            label.setText(f"Input device: {'active' if status else 'inactive'}")
            self._status[0] = status
            if status:
                # Reset watchdog timer, since it is under our control
                self._input_device_watchdog_timer.reset()
        elif "Mindrove" in label.text(): 
            label.setText(f"Mindrove: {'active' if status else 'inactive'}")
            if status:
                # Reset watchdog timer, since it is under our control
                self._mindrove_watchdog_timer.reset()
            self._status[1] = status
        elif "Plux" in label.text(): 
            label.setText(f"Plux: {'active' if status else 'inactive'}")
            if status:
                # Reset watchdog timer, since it is under our control
                self._plux_watchdog_timer.reset()
            self._status[2] = status

    def _check_statuses(self):
        if all(self._status):
            self._node.get_logger().info("All items are active, starting logging")
            if not self._log_manager.cycle_logging(name="manual_exercise"):
                self._node.get_logger().info("Failed to start logging")
                return
            
            # Cancel this timer once logging begins
            self._logger_timer.cancel()

class ExperimentControlGui(URControlQtWindow):
    def __init__(self, node):
        super().__init__(node)

        self._tool_watchdog = QLabel(text="Tool: inactive")
        self._input_device_watchdog = QLabel(text="Input device: inactive")
        self._mindrove_watchdog = QLabel(text="Mindrove: inactive")
        self._plux_watchdog = QLabel(text="Plux: inactive")

        self._rviz_manager = RvizManager(
            node=node,
            num_repetitions=_NUM_REPETITIONS,
            refresh_rate=1.0/120.0,
            move_camera_with_exercise=True,
            align_with_path=False)

        self._robot_manager = RobotControlArbiter(
            node=node,
            robot=self._robot, 
            simulate_tool=not _USE_TOOL,
            num_repetitions=_NUM_REPETITIONS,
            engagement_debounce=0.5,
            state_change_callback=self._refresh_ui,
            experiment_feedback_callback=self._rviz_manager.exercise_feedback,
            experiment_repetition_completion_callback=self._rviz_manager.exercise_completion_callback,
            experiment_set_completion_callback=self._rviz_manager.reset,
            watchdog_input_device_change_callback=partial(self._update_watchdog, self._input_device_watchdog),
            watchdog_tool_change_callback=partial(self._update_watchdog, self._tool_watchdog))
        
        self._exercise_manager = ExerciseManager(
            node=node,
            distance_threshold=_DISTANCE_THRESHOLD,
            pose_callback=self._rviz_manager.update_pose)

        self._mindrove_sub = self._node.create_subscription(
            msg_type=MindroveArmBandEightChannelMsg,
            topic=MINDROVE_ROS_TOPIC_NAME,
            callback=partial(self._update_watchdog, self._mindrove_watchdog),
            qos_profile=get_realtime_qos_profile()
        )

        self._plux_sub = self._node.create_subscription(
            msg_type=PluxMsg,
            topic=PLUX_ROS_TOPIC_NAME,
            callback=partial(self._update_watchdog, self._plux_watchdog),
            qos_profile=get_realtime_qos_profile()
        )

        self._mindrove_watchdog_timer = self._node.create_timer(
            1.0,
            callback=partial(self._update_watchdog, self._mindrove_watchdog, False)
        )

        self._plux_watchdog_timer = self._node.create_timer(
            1.0,
            callback=partial(self._update_watchdog, self._plux_watchdog, False)
        )

        self._log_manager = RosbagManager(
            node=self._node
        )

        exercise_tab_layout = QVBoxLayout()

        self._exercise_tab_label = QLabel(f"State: {RobotControlStatus.UNINITIALIZED.name}")
        labels = (
            self._exercise_tab_label,
            self._tool_watchdog,
            self._input_device_watchdog,
            self._mindrove_watchdog,
            self._plux_watchdog
        )
        
        for label in labels:
            exercise_tab_layout.addWidget(label)
        
        self._buttons = {
            QPushButton("INITIALIZE ROBOT", self): self._robot_manager.start_robot_program,
            QPushButton("STOP ROBOT", self): self._robot_manager.stop_experiment_override,
            # TODO(george): call RViz visualizer to set handedness here...
            QPushButton("DEPLOY ROBOT LHS", self): partial(self._robot_manager.move_to_home, _LHS_JOINT_ANGLES),
            QPushButton("DEPLOY ROBOT RHS", self): partial(self._robot_manager.move_to_home, _RHS_JOINT_ANGLES),
            QPushButton("RESET ARM", self): self._robot_manager.reset_arm_pose,
            QPushButton("TRANSLATE END EFFECTOR", self): self.__translate_end_effector,
            QPushButton("PLAN BICEP CURL", self): partial(self.__exercise_planning, ExerciseType.VERTICAL_BICEP_CURL),
            QPushButton("PLAN LATERAL RAISE", self): partial(self.__exercise_planning, ExerciseType.VERTICAL_LATERAL_RAISE),
            QPushButton("STOP PLANNING", self): self.__stop_planning,
            QPushButton("LOAD EXERCISE", self): self.__load_exercise,
            QPushButton("PREVIEW EXERCISE", self): self.__preview_exercise,
            QPushButton("SET EXERCISE", self): self.__set_exercise,
            QPushButton("SAVE EXERCISE", self): self.__save_exercise
        }

        self._last_state = RobotControlStatus.UNINITIALIZED

        self._create_tab(name="Exercise Tab", layout=exercise_tab_layout, tab_create_func=self._exercise_tab_constructor)

    @property
    def sensor_watchdog(self):
        return 'inactive' in self._plux_watchdog.text() or 'inactive' in self._mindrove_watchdog.text()
        
    def _refresh_ui(self, state: RobotControlStatus):
        self._exercise_tab_label.setText(f"State: {state.name}")

        self._last_state = state
        
        # TODO(george): enable specific buttons based upon current state
        # for button in self._buttons.keys():
        #     match state:
        #         case RobotControlStatus.UNINITIALIZED:
        #             if button.text() == "INITIALIZE ROBOT": button.setEnabled(True)
        #             else: button.setEnabled(False)
        #         case RobotControlStatus.INITIALIZED:
        #             if "DEPLOY ROBOT" in button.text() or button.text() == "STOP ROBOT": button.setEnabled(True)
        #             else: button.setEnabled(False)
        #         case RobotControlStatus.DEPLOYING:
        #             button.setEnabled(False)
        #         case RobotControlStatus.IDLE:
        #             if button.text() in ("INITIALIZE ROBOT") or "DEPLOY ROBOT":
        #                 pass
        #         case RobotControlStatus.ACTIVE:
        #             if not self.sensor_watchdog:
        #                 # Stop the experiment if the sensor watchdog has been tripped
        #                 self._robot_manager.stop_experiment_override()
                

    def _update_watchdog(self, label: QLabel, status: bool | PluxMsg | MindroveArmBandEightChannelMsg):
        if "Tool" in label.text(): 
            label.setText(f"Tool: {'active' if status else 'inactive'}")
        elif "Input" in label.text(): 
            label.setText(f"Input device: {'active' if status else 'inactive'}")
        elif "Mindrove" in label.text(): 
            label.setText(f"Mindrove: {'active' if status else 'inactive'}")
            if status:
                # Reset watchdog timer, since it is under our control
                self._mindrove_watchdog_timer.reset()
        elif "Plux" in label.text(): 
            label.setText(f"Plux: {'active' if status else 'inactive'}")
            if status:
                # Reset watchdog timer, since it is under our control
                self._plux_watchdog_timer.reset()

    def _exercise_tab_constructor(self, layout: QLayout):
        for button, callback_func in self._buttons.items():
            button.clicked.connect(callback_func)
            layout.addWidget(button)

    def __translate_end_effector(self):
        if not self._robot_manager.backdrive_robot(
            # Freedrive the robot in the translational axes
            SetFreedriveParams.Request(
                type=SetFreedriveParams.Request.TYPE_STRING,
                free_axes=[True, True, True, False, False, False],
                feature_constant=SetFreedriveParams.Request.FEATURE_TOOL
            )
        ):
            self._node.get_logger().error("Failed to enter freedrive")
            return

    def __exercise_planning(self, exercise_type: ExerciseType):
        if not self._robot_manager.backdrive_robot(
            SetFreedriveParams.Request(
                type=SetFreedriveParams.Request.TYPE_STRING,
                free_axes=self._exercise_manager.get_exercise_dofs(exercise_type),
                feature_constant=SetFreedriveParams.Request.FEATURE_TOOL
            )
        ):
            self._node.get_logger().error("Failed to enter freedrive")
            return
        
        self._exercise_manager.start_monitoring(name=postfix_string_with_current_time(exercise_type.name))

    def __stop_planning(self):
        if self._exercise_manager.is_monitoring():
            self._exercise_manager.stop_monitoring()
        else:
            if current_position := self._exercise_manager.get_last_joint_state():
                self._robot_manager.set_position_as_home(current_position)
        self._robot_manager.exit_planning()

    def __preview_exercise(self):
        signal = threading.Event()
        if self._robot_manager.preview_exercise(signal):
            signal.set()

            _dialog = QDialog()
            _dialog.setLayout(QVBoxLayout())
            _dialog.layout().addWidget(QLabel("Close this window to stop the preview. The robot will return to the start."))
            _dialog.exec_()

            signal.clear()
        else:
            self._node.get_logger().error("Failed to preview exercise")

    def __load_exercise(self):
        options = QFileDialog.Options()
        fp, _ = QFileDialog.getOpenFileName(
            self, 
            "Select exercise trajectory", 
            "", 
            "Exercise file (*.exercise)", 
            options = options
        )

        if fp and self._exercise_manager.load_exercise(fp):
            self._node.get_logger().info("Loaded exercise successfully")
        else:
            self._node.get_logger().info("No input exercise selected or failed to load exercise")

    def __set_exercise(self):
        if exercise := self._exercise_manager.get_exercise():
            if not self._robot_manager.load_experiment(
                exercise=exercise
            ):
                self._node.get_logger().error("Failed to set experiment")
            else:
                # Visualize the exercise once it has been set
                self._rviz_manager.visualize_exercise(exercise=exercise)

                if not self._log_manager.cycle_logging(name=exercise.name):
                    self._node.get_logger().error("Failed to start logging...")

    def __save_exercise(self):
        if exercise := self._exercise_manager.get_exercise():
            pass
        else:
            return
        
        options = QFileDialog.Options()
        fp, _ = QFileDialog.getSaveFileName(
            self, 
            "Save exercise trajectory", 
            f"{exercise.name.lower()}.exercise", 
            "Exercise file (*.exercise)", 
            options = options
        )

        if fp and self._exercise_manager.save_exercise(fp):
            self._node.get_logger().info(f"Saved exercise to {fp}")
        else:
            self._node.get_logger().info("No output selected or error saving exercise, file not saved")

def main(args=None):
    # Create the application
    app = QApplication(sys.argv)

    rclpy.init(args=args)

    node = Node("exercise_primary_node")
    if node.declare_parameter('manual', False).value:
        # Create the main window
        main_window = ManualExperimentControlGui(node)
        main_window.show()

        executor = MultiThreadedExecutor()
        executor.add_node(node)

        rclpy_spinner = RclpySpinner([node])
        rclpy_spinner.start()
    else:
        # Create the main window
        main_window = ExperimentControlGui(node)
        main_window.show()

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.add_node(main_window._robot._node)
        executor.add_node(main_window._robot._service_node)

        rclpy_spinner = RclpySpinner([node, main_window._robot._node, main_window._robot._service_node])
        rclpy_spinner.start()
    
    # Run the application's event loop
    app.exec_()
    rclpy_spinner.quit()
    sys.exit()