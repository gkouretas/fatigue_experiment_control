from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

_hard_coded_args = {
    "ur_type": "ur10e",
    "robot_ip": "192.168.1.102",
    "launch_simulation": "false",
    "headless_mode": "true"
}

def generate_launch_description():
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ur_robot_driver"), "/launch/ur_control.launch.py"]),
        launch_arguments = _hard_coded_args.items()
    )

    qt_launch = Node(
        package="fatigue_experiment_control",
        executable="experiment_gui",
        parameters=[{"manual": False}]
    )

    mindrove_launch = Node(
        package="ros2_mindrove",
        executable="pub"
    )

    plux_launch = Node(
        package="ros2_plux_biosignals",
        executable="pub"
    )

    # return LaunchDescription(
    #     [ur_control_launch, qt_launch, mindrove_launch, plux_launch]
    # )
    # Remove plux b/c bluetooth isn't working on the system
    return LaunchDescription(
        [ur_control_launch, qt_launch, mindrove_launch]
    )