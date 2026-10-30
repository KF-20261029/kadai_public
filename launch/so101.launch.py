"""Start the SO-101 scene with standard ROS 2 control components."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro


def launch_setup(context):
    project_root = Path(LaunchConfiguration("project_root").perform(context)).resolve()
    controller_file = str(project_root / "config/controllers.yaml")
    controller_name = LaunchConfiguration("controller").perform(context)
    description = xacro.process_file(
        str(project_root / "description/so101.urdf.xacro"),
        mappings={"project_root": str(project_root), "headless": "true"},
    ).toxml()

    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": ParameterValue(description, value_type=str),
            "use_sim_time": True,
        }],
        output="screen",
    )
    simulator = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        parameters=[{"use_sim_time": True}, controller_file,
                    str(project_root / "config/cameras.yaml"),
                    str(project_root / "config/pen_states.yaml")],
        additional_env={"DISPLAY": ""},
        output="screen",
        on_exit=Shutdown(),
    )
    controllers = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster", controller_name,
            "--param-file", controller_file,
            "--controller-manager-timeout", "30",
        ],
        output="screen",
    )
    return [state_publisher, simulator, controllers]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value=str(Path(__file__).resolve().parents[1]),
            description="Directory containing the task scene and ROS configuration.",
        ),
        DeclareLaunchArgument(
            "controller",
            default_value="position_controller",
            choices=["position_controller", "trajectory_controller"],
            description="Controller used for the SO-101 joints.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
