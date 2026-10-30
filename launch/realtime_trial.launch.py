"""Record a trial that keeps the ROS-controlled simulation running."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    recording = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '--use-sim-time', '-s', 'mcap',
             '--storage-preset-profile', 'zstd_fast', '-o', LaunchConfiguration('output'), '--topics',
             '/mujoco_camera_plugin/overview/color', '/joint_states',
             '/free_joint_state_publisher/free_joint_states', '/clock'],
        output='screen', sigterm_timeout='120',
    )
    policy = Node(
        package='so101_policy', executable='execute_realtime',
        parameters=[{'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int),
                     'duration_seconds': ParameterValue(LaunchConfiguration('duration_seconds'), value_type=float),
                     'lead_seconds': ParameterValue(LaunchConfiguration('lead_seconds'), value_type=float),
                     'adapter': LaunchConfiguration('adapter'),
                     'metrics_path': PathJoinSubstitution([LaunchConfiguration('output'), 'metrics.json'])}],
        output='screen',
    )
    return LaunchDescription([
        DeclareLaunchArgument('output', default_value='/workspace/outputs/realtime_trial'),
        DeclareLaunchArgument('seed', default_value='7100'),
        DeclareLaunchArgument('adapter', default_value='/workspace/models/pen_lora'),
        DeclareLaunchArgument('duration_seconds', default_value='75.0'),
        DeclareLaunchArgument('lead_seconds', default_value='0.35'),
        RegisterEventHandler(OnProcessExit(target_action=recording,
                             on_exit=[EmitEvent(event=Shutdown(reason='Recording exited.'))])),
        RegisterEventHandler(OnProcessExit(target_action=policy,
                             on_exit=[EmitEvent(event=Shutdown(reason='Policy exited.'))])),
        recording,
        TimerAction(period=3.0, actions=[policy]),
    ])
