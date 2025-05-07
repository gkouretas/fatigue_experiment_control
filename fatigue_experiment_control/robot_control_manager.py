import threading

from enum import IntEnum, auto
from typing import Callable

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.action import ActionClient
from rclpy.client import Future

from ur_msgs.msg import ToolDataMsg
from idl_definitions.msg import UserInputMsg
from pi_user_input_node.user_input_node_config import USER_INPUT_QOS_PROFILE, USER_INPUT_TOPIC_NAME
from ur10e_custom_control.ur_robot_sm import URRobotSM
from ur_msgs.action import DynamicForceModePath
from ur_dashboard_msgs.msg import RobotMode, SafetyMode
from ur_msgs.srv import SetIO, SetFreedriveParams
from ur_dashboard_msgs.action import SetMode
from ur10e_custom_control.ur10e_typedefs import URService
from ur10e_custom_control.ur10e_configs import UR_QOS_PROFILE
from control_msgs.action import FollowJointTrajectory
from fatigue_experiment_control.exercise_manager import Exercise
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Wrench, Twist, Vector3
from nav_msgs.msg import Path

from dataclasses import dataclass, field

# Active threshold in [V] for tool analog input
# NOTE(george): this would preferably be a digital input, but we only get access to an analog input 
_ANALOG_ACTIVE_THRESHOLD = 2.5
_SPEED_LIMITS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
_CARTESIAN_DEVIATION_LIMIT = 1.0
_ANGULAR_DEVIATION_LIMIT = 1.0

@dataclass
class ToolState:
    is_on: bool
    is_engaged: bool

    # Initialize watchdog to False
    watchdog: bool = field(default=False, init=False)

@dataclass
class InputState:
    is_active: bool

    # Initialize watchdog to False
    watchdog: bool = field(default=False, init=False)

class RobotControlStatus(IntEnum):
    ERROR = auto()
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    IDLE = auto()
    TRAJECTORY = auto()
    PLANNING = auto()
    PREVIEW = auto()
    READY = auto()
    ACTIVE = auto()
    PAUSED = auto()

class RobotControlArbiter:
    def __init__(self, 
                 node: Node, 
                 robot: URRobotSM, 
                 engagement_debounce: float | None = None, 
                 state_change_callback: Callable[[RobotControlStatus], None] = None,
                 deploy_feedback_callback: Callable[[FollowJointTrajectory.Feedback], None] = None,
                 experiment_feedback_callback: Callable[[DynamicForceModePath.Feedback], None] = None,
                 watchdog_input_device_change_callback: Callable[[bool], None] = None,
                 watchdog_tool_change_callback: Callable[[bool], None] = None):
        """
        RobotControlArbiter

        This node is responsible for being the middle-man for all UR related requests for fatigue experiments.

        It performs actions in a state-machine like manner, where requests come in and are serviced as a function of the robot's current state.

        Args:
            node (Node): _description_
            robot (URRobotSM): _description_
            engagement_debounce (float | None, optional): _description_. Defaults to None.
            state_change_callback (Callable[[RobotControlStatus], None], optional): _description_. Defaults to None.
        """
        self._state = RobotControlStatus.UNINITIALIZED
        self._robot_mutex = threading.RLock()
        self._telemetry_mutex = threading.RLock()

        self._home_pose = None

        self._node = node
        self._robot = robot

        self._state_change_callback = state_change_callback
        self._deploy_feedback_callback = deploy_feedback_callback
        self._experiment_feedback_callback = experiment_feedback_callback
        self._watchdog_input_device_change_callback = watchdog_input_device_change_callback
        self._watchdog_tool_change_callback = watchdog_tool_change_callback

        self._is_active = False

        self._pending_trajectory_state: RobotControlStatus | None = None
        self._previous_state: RobotControlStatus = RobotControlStatus.UNINITIALIZED
        
        self._pending_exercise: Exercise | None = None
        self._tool_state = ToolState(is_on=False, is_engaged=False)
        self._input_device_state = InputState(is_active=False)

        # Subscribe to relevant control topics
        self._tool_status = self._node.create_subscription(
            msg_type=ToolDataMsg,
            topic="/io_and_status_controller/tool_data",
            callback=self._update_tool_state,
            qos_profile=UR_QOS_PROFILE
        )

        self._input_device_status = self._node.create_subscription(
            msg_type=UserInputMsg,
            topic=USER_INPUT_TOPIC_NAME,
            qos_profile=USER_INPUT_QOS_PROFILE,
            callback=self._update_user_input_state
        )

        # Watchdog monitors for tool state / input device state
        self._tool_state_watchdog = self._node.create_timer(
            1.0, 
            self._trip_tool_state_watchdog,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self._input_device_state_watchdog = self._node.create_timer(
            1.0, 
            self._trip_input_device_watchdog,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Debounce timer for tool engagement
        if engagement_debounce is not None:
            self._user_input_debounce = self._node.create_timer(
                engagement_debounce,
                self._trip_debounce,
                callback_group=MutuallyExclusiveCallbackGroup()
            )

            #self._user_input_debounce.cancel()
            self._debounce_active = False
            self._has_debounce_been_run = False
            self._debounce_triggered = False
        else:
            self._debounce_active = False
            self._user_input_debounce = None
            self._has_debounce_been_run = False

            # Always true
            self._debounce_triggered = True

        # Cyclic monitor function
        self._cyclic_monitor_timer = self._node.create_timer(
            0.01,
            self.monitor_experiment,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

    @property
    def _debounce_timer_running(self) -> bool:
        """
        Utility property for checking whether the debounce timer is active

        Returns:
            bool: true if running, false otherwise
        """
        return self._debounce_active
        
    @property
    def is_ready(self) -> bool:
        """
        Checks if the user is ready to execute the test

        Returns:
            bool: true if the user is ready, false otherwise
        """
        with self._telemetry_mutex:            
            return self._input_device_state.is_active and self._input_device_state.watchdog
        
    @property
    def tool_engaged(self):
        """
        Checks if the user has engaged the tool

        Returns:
            bool: true if the user has engaged the tool, false otherwise
        """
        # NOTE: this is a bit of a messy debounce scheme, may want to tweak if this doesn't work as intended into a less
        # convoluted implementation
        # 
        # The tool is engaged if one of the two conditions are met:
        # 1) the raw tool state is engaged and our debounce is not running (signaling the "true" state is disengaged)
        # 2) the raw tool state is disengaged and our debounce timer is active (signaling the "true" state is engaged)
        #    - In addition, the debounce must have been run since the initial state is disengaged
        with self._telemetry_mutex:
            is_engaged = (self._tool_state.is_engaged and not self._debounce_timer_running) or \
                    (self._debounce_timer_running and not self._tool_state.is_engaged and self._has_debounce_been_run)

            return is_engaged and self._tool_state.watchdog
        
    def _change_state(self, state: RobotControlStatus) -> None:
        """
        Faciliates a state change, where the internal state is changes and the external callback is performed (if provided).
        
        No state change logic exists here, as that is in the `_run_state_machine`.

        Args:
            state (RobotControlStatus): Updated state
        """
        self._node.get_logger().info(f"State: {self._state.name} -> {state.name}")
        self._previous_state = self._state
        self._state = state
        if self._state_change_callback is not None:
            self._state_change_callback(state)
        
    def start_robot_program(self) -> bool:
        """
        Start the robot program and initialize the tool

        Returns:
            bool: true if the robot program successfully started, false otherwise
        """
        with self._robot_mutex:
            if self._state > RobotControlStatus.UNINITIALIZED:
                return False
            
            request = self._robot.request_mode(
                SetMode.Goal(
                    target_robot_mode=RobotMode.RUNNING,
                    play_program=True,
                    stop_program=False
                ),
                blocking=True
            )

            if request is not None and request.result.success:
                pass
            else:
                self._change_state(RobotControlStatus.ERROR)
                return False

            # Set tool voltage to 12V
            request = self._robot.call_service(
                URService.IOAndStatusController.SRV_SET_IO,
                request=SetIO.Request(fun=SetIO.Request.FUN_SET_TOOL_VOLTAGE, 
                                        state=float(SetIO.Request.STATE_TOOL_VOLTAGE_12V))
            )

            if request is not None and request.success:
                pass
            else:
                self._change_state(RobotControlStatus.ERROR)
                return False

            self._change_state(RobotControlStatus.INITIALIZED)
            return True

    def stop_robot_program(self) -> bool:
        """
        Stop the robot program and power off the tool

        Returns:
            bool: true if the robot program was successfully powered off, false otherwise
        """
        with self._robot_mutex:
            if self._state == RobotControlStatus.UNINITIALIZED:
                return False
            
            # Power the tool off
            request = self._robot.call_service(
                URService.IOAndStatusController.SRV_SET_IO,
                request=SetIO.Request(fun=SetIO.Request.FUN_SET_TOOL_VOLTAGE, 
                                        state=float(SetIO.Request.STATE_TOOL_VOLTAGE_0V))
            )

            if request is not None and request.success:
                pass
            else:
                self._change_state(RobotControlStatus.ERROR)
                return False

            self._robot.request_mode(
                SetMode.Goal(
                    target_robot_mode=RobotMode.POWER_OFF,
                    play_program=False,
                    stop_program=False
                ),
                blocking=True
            )

            if request is not None and request.success:
                pass
            else:
                self._change_state(RobotControlStatus.ERROR)
                return False

            self._change_state(RobotControlStatus.UNINITIALIZED)
            return True

    def backdrive_robot(self, params: SetFreedriveParams.Request) -> bool:
        """
        Backdrive the robot in freedrive control mode.

        Args:
            request (SetFreedriveParams.Request): Freedrive control request.

        Returns:
            bool: true if able to successfully start freedrive, false otherwise
        """
        with self._robot_mutex:
            if self._state != RobotControlStatus.IDLE or not self._robot.program_running:
                self._node.get_logger().error(f"Invalid internal state: {self._state.name} or program is not running")
                return False
            
            if result := self._robot.call_service(
                URService.FreedriveController.SRV_SET_FREEDRIVE_PARAMS,
                request=params
            ): 
                if result.success:
                    self._robot.run_freedrive_control()
                    self._change_state(RobotControlStatus.PLANNING)
                    return True
                else:
                    self._node.get_logger().error(f"Failed result: {result}")
                    return False
            else:
                return False
            
    def exit_planning(self):
        with self._robot_mutex:
            self._robot.disable_freedrive()
            self._robot.stop_freedrive_control()
            self._change_state(RobotControlStatus.IDLE)

    def move_to_home(self, home: list[list[float]]):
        self._home_pose = home
        self._move_to(self._home_pose)

    def reset_arm_pose(self):
        if self._home_pose is not None:
            self._move_to(self._home_pose)

    def preview_exercise(self, signal: threading.Event):
        def cycle_trajectory():
            signal.wait()

            future: Future | None = None
            while self._pending_exercise is not None and \
                self._state == RobotControlStatus.PREVIEW and \
                    signal.is_set():
                        if self._tool_state.is_on and self._tool_state.is_engaged:
                            self._pause_program()
                            return
                    
                        if future is None or future.done():
                            # Skip first iteration since duration @ t0 = 0
                            future = self._robot.send_trajectory(
                                self._pending_exercise.joint_angles[1:] if not reverse else self._pending_exercise.joint_angles[::-1][1:],
                                self._pending_exercise.duration[1:], # NOTE: can't reverse duration since it is sequential...
                                blocking=False
                            )

                            if not isinstance(future, Future):
                                msg=f"Unexpected type: {future}"
                                self._node.get_logger().info(msg)
                                self.stop_robot_program()
                                raise ValueError(msg)
                
                            reverse = not reverse

            if not signal.is_set():
                # Return to the starting position
                self._robot.send_trajectory(
                    [self._pending_exercise.joint_angles[0]],
                    [Duration(sec=10.0)],
                    blocking=True
                )

            return

        if self._pending_exercise is not None and (self._state == RobotControlStatus.IDLE):
            self._change_state(RobotControlStatus.PREVIEW)
            threading.Thread(target=cycle_trajectory, daemon=True).start()
            return True
        
        return False

    def _handle_trajectory_future(self, ac_client: ActionClient, result_callback: Callable | None, future: Future):
        result = future.result()
        if result is not None and result.accepted:
            self._node.get_logger().info("Trajectory accepted")
            result_future = ac_client._get_result_async(future.result())
            if result_callback is not None:
                result_future.add_done_callback(result_callback)
        else:
            self._node.get_logger().error(f"Trajectory failure: {future.result()}")
            self._change_state(RobotControlStatus.ERROR)

    def deploy_robot(self, waypoints: list[list[float]], durations: list[Duration]):
        with self._robot_mutex:
            self._robot.set_action_completion_callback(self._handle_trajectory_future)
            self._robot.set_action_result_callback(self._robot_control_completion)
            
            self._robot.send_trajectory(
                waypts=waypoints,
                time_vec=durations,
                blocking=False
            )

    def _move_to(self, pose: list[float], duration: Duration = Duration(sec=10)):
        self._robot.set_action_completion_callback(self._handle_trajectory_future)
        self._robot.set_action_result_callback(self._robot_control_completion)

        self._robot.send_trajectory(
            [pose],
            [duration],
            blocking=False
        )

    def _robot_control_completion(self, *args):
        self._change_state(RobotControlStatus.IDLE)

    def _trip_tool_state_watchdog(self):
        with self._telemetry_mutex:
            if self._tool_state.watchdog:
                self._tool_state.watchdog = False
                if self._watchdog_tool_change_callback is not None:
                    self._watchdog_tool_change_callback(self._tool_state.watchdog)
                self._node.get_logger().warning("Tool state watchdog tripped!")

    def _trip_input_device_watchdog(self):
        with self._telemetry_mutex:
            if self._input_device_state.watchdog:
                self._input_device_state.watchdog = False
                if self._watchdog_input_device_change_callback is not None:
                    self._watchdog_input_device_change_callback(self._input_device_state.watchdog)

                self._node.get_logger().warning("Input device state watchdog tripped!")

    def _trip_debounce(self):
        with self._telemetry_mutex:
            if self._debounce_active:
                self._debounce_triggered = True
                self._has_debounce_been_run = True
                self._debounce_active = False
                self._node.get_logger().info(f"Debounce tripped -> {self.tool_engaged}")
                #self._user_input_debounce.cancel()

    def load_experiment(self, exercise: Exercise) -> bool:
        with self._telemetry_mutex:
            if self._state == RobotControlStatus.IDLE:
                self._exercise = exercise
                self._pending_trajectory_state = RobotControlStatus.READY
                self._move_to(self._exercise.joint_angles[0])
                return True
            else:
                return False

    def stop_experiment_override(self):
        with self._robot_mutex:
            self._robot.stop_robot()
            self._change_state(RobotControlStatus.UNINITIALIZED)

    def _start_experiment(self, exercise: Exercise) -> bool:
        with self._robot_mutex:
            self._robot.set_action_completion_callback(self._handle_trajectory_future)
            #self._robot.set_action_feedback_callback(self._experiment_feedback)
            self._robot.set_action_result_callback(self._robot_control_completion)
            
            goal = DynamicForceModePath.Goal(
                task_frame=exercise.poses[0],
                wrench_baseline=Wrench(force=Vector3(x=0.0, y=0.0, z=0.0),
                                        torque=Vector3(x=0.0, y=0.0, z=0.0)),
                type=1,
                speed_limits=Twist(
                    linear=Vector3(x=_SPEED_LIMITS[0], y=_SPEED_LIMITS[1], z=_SPEED_LIMITS[2]),
                    angular=Vector3(x=_SPEED_LIMITS[3], y=_SPEED_LIMITS[4], z=_SPEED_LIMITS[5])),
                deviation_limits=[
                    _CARTESIAN_DEVIATION_LIMIT,
                    _CARTESIAN_DEVIATION_LIMIT,
                    _CARTESIAN_DEVIATION_LIMIT,
                    _ANGULAR_DEVIATION_LIMIT,
                    _ANGULAR_DEVIATION_LIMIT,
                    _ANGULAR_DEVIATION_LIMIT
                ],
                force_mode_path=Path(poses=exercise.poses),
                wrench_path_kp=0.0,
                wrench_path_ki=0.0,
                wrench_path_kd=0.0,
                wrench_path_max_force_magnitude=0.0,
                wrench_path_max_torque_magnitude=0.0,
                waypoint_tolerances=[0.025,0.025,0.025,0.025,0.025,0.025],
                compliance_tolerances=[
                    DynamicForceModePath.Goal.ALWAYS_INACTIVE,
                    DynamicForceModePath.Goal.ALWAYS_ACTIVE,
                    DynamicForceModePath.Goal.ALWAYS_INACTIVE,
                    _ANGULAR_DEVIATION_LIMIT,
                    _ANGULAR_DEVIATION_LIMIT,
                    _ANGULAR_DEVIATION_LIMIT
                ],        
            )

            self._robot.run_dynamic_force_mode(goal, blocking=False)

    def _pause_program(self) -> bool:
        with self._robot_mutex:
            status = False
            if result := self._robot.call_service(URService.DashboardClient.SRV_PAUSE):
                status = result.success
                if status:
                    self._change_state(RobotControlStatus.PAUSED)

            return status

    def _play_program(self) -> bool:
        with self._robot_mutex:
            status = False
            if result := self._robot.call_service(URService.DashboardClient.SRV_PLAY):
                status = result.success
                if status:
                    self._change_state(RobotControlStatus.IDLE)

            return status

    _pause_experiment = _pause_program
    _resume_experiment = _play_program
    
    def _run_state_machine(self):
        with self._robot_mutex:
            match self._robot.current_mode.mode:
                case RobotMode.RUNNING | RobotMode.IDLE:
                    if self._state < RobotControlStatus.INITIALIZED and self._robot.program_running:
                        # If the program is running and we don't think it is, stop it
                        self._node.get_logger().warning(f"Unexpected current mode {self._robot.current_mode} in local state {self._state.name}")
                        self.stop_robot_program()
                case _:
                    if self._state != RobotControlStatus.UNINITIALIZED and self._state != RobotControlStatus.ERROR:
                        self._node.get_logger().info(f"Unhandled mode {self._robot.current_mode}, changing to uninit")
                        self._change_state(RobotControlStatus.UNINITIALIZED)
                
            match self._robot.current_safety_mode.mode:
                case SafetyMode.NORMAL:
                    pass
                case _:
                    if self._state > RobotControlStatus.UNINITIALIZED:
                        self._change_state(RobotControlStatus.ERROR)

            match self._state:
                case RobotControlStatus.UNINITIALIZED | RobotControlStatus.INITIALIZED:
                    pass
                case RobotControlStatus.IDLE:
                    if self.is_ready:
                        self._change_state(state=RobotControlStatus.READY)
                case RobotControlStatus.TRAJECTORY:
                    if self.tool_engaged:
                        self._change_state(RobotControlStatus.IDLE)
                case RobotControlStatus.PLANNING:
                    self._robot.ping_freedrive()
                    # if self.tool_engaged:
                    #     # Ping freedrive if the tool is engaged
                    #     self._robot.ping_freedrive()
                    # else:
                    #     # Disable freedrive if the tool is disengaged
                    #     self._robot.disable_freedrive()
                    #     self._change_state(RobotControlStatus.IDLE)
                case RobotControlStatus.READY:
                    if self.is_ready and self.tool_engaged and self._pending_exercise is None:
                        # Attempt to start the experiment
                        if self._start_experiment(self._pending_exercise):
                            self._change_state(state=RobotControlStatus.ACTIVE)
                        else:
                            self._change_state(state=RobotControlStatus.ERROR)
                case RobotControlStatus.ACTIVE:
                    if not self.is_ready:
                        # If we are not ready, pause the experiment
                        if self._pause_experiment():
                            self._change_state(state=RobotControlStatus.PAUSED)
                        else:
                            self._change_state(state=RobotControlStatus.ERROR)
                case RobotControlStatus.PAUSED:
                    match self._previous_state:
                        case RobotControlStatus.ACTIVE:
                            cond = self.is_ready and self.tool_engaged
                        case RobotControlStatus.TRAJECTORY:
                            cond = not self.tool_engaged
                        case _:
                            cond = False

                    if cond:
                        if self._play_program():
                            self._change_state(state=self._previous_state)
                        else:
                            self._change_state(state=RobotControlStatus.ERROR)
                case _:
                    pass

    def monitor_experiment(self):
        with self._robot_mutex:
            self._run_state_machine()

    def _update_tool_state(self, msg: ToolDataMsg):
        previous_state = self._tool_state.is_engaged
        self._tool_state.is_on = msg.tool_output_voltage > 0.0 # Check if tool voltage is non-zero
        self._tool_state.is_engaged = msg.analog_input2 > _ANALOG_ACTIVE_THRESHOLD # Check if our analog voltage is above the "active" threshold
        self._tool_state_watchdog.reset()

        if self._user_input_debounce is not None:
            # Manage debounce timer
            if previous_state != self._tool_state.is_engaged: 
                # If a state change is detected, check if the debounce timer is running
                if not self._debounce_timer_running: 
                    # Start debounce timer
                    self._node.get_logger().info("Starting debounce")
                    self._user_input_debounce.reset()
                    self._debounce_active = True
                else:
                    # Reset debounce timer if jitter occurs before timer is tripped
                    self._node.get_logger().info(f"Jitter detected: {previous_state} -> {self._tool_state.is_engaged}")
                    self._user_input_debounce.reset()

        if not self._tool_state.watchdog:
            self._tool_state.watchdog = True
            if self._watchdog_tool_change_callback is not None:
                self. _watchdog_tool_change_callback(self._tool_state.watchdog)
            self._node.get_logger().info("Established communication with tool state")

    def _update_user_input_state(self, msg: UserInputMsg):
        self._input_device_state.is_active = msg.is_active
        self._input_device_state_watchdog.reset()

        if not self._input_device_state.watchdog:
            self._input_device_state.watchdog = True
            if self._watchdog_input_device_change_callback is not None:
                self. _watchdog_input_device_change_callback(self._input_device_state.watchdog)
            self._node.get_logger().info("Established communication with input device state")