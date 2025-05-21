from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration

from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

_hard_coded_args = {
    "ur_type": "ur10e",
    "robot_ip": "192.168.56.101",
    "launch_simulation": "true",
    "simulation_version": "latest",
    "headless_mode": "true"
}

def generate_launch_description():
    qt_launch = Node(
        package="fatigue_experiment_control",
        executable="experiment_gui",
        parameters=[{"manual": False}]
    )

    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ur_robot_driver"), "/launch/ur_control.launch.py"]),
        launch_arguments = _hard_coded_args.items()
    )

    return LaunchDescription(
        [ur_control_launch, qt_launch]
    )