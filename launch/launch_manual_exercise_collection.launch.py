from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

def generate_launch_description():
    qt_launch = Node(
        package="fatigue_experiment_control",
        executable="experiment_gui",
        parameters=[{"manual": True}]
    )

    mindrove_launch = Node(
        package="ros2_mindrove",
        executable="pub"
    )

    plux_launch = Node(
        package="ros2_plux_biosignals",
        executable="pub"
    )

    # Remove plux b/c bluetooth isn't working on the system
    return LaunchDescription(
        [qt_launch, mindrove_launch]
    )