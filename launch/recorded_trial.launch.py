"""Run and record a trained-policy trial with standard ROS 2 tools."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    recording = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '--use-sim-time', '-s', 'mcap',
            '--storage-preset-profile', 'zstd_fast',
            '-o', LaunchConfiguration('output'), '--topics',
            '/mujoco_camera_plugin/overview/color', '/joint_states',
            '/free_joint_state_publisher/free_joint_states', '/clock',
        ],
        output='screen',
        sigterm_timeout='120',
    )
    policy = Node(
        package='so101_policy', executable='execute_chunk',
        parameters=[{'chunks': ParameterValue(LaunchConfiguration('chunks'), value_type=int),
                     'seed': ParameterValue(LaunchConfiguration('seed'), value_type=int),
                     'use_lora': ParameterValue(LaunchConfiguration('use_lora'), value_type=bool),
                     'adapter': LaunchConfiguration('adapter')}],
        output='screen',
    )
    return LaunchDescription([
        DeclareLaunchArgument('output', default_value='/workspace/outputs/pen_trial_02',
                              description='Output directory for the recorded ROS data.'),
        DeclareLaunchArgument('chunks', default_value='45',
                              description='Number of 50-action policy chunks.'),
        DeclareLaunchArgument('seed', default_value='-1',
                              description='Inference random seed, or -1 for the current generator state.'),
        DeclareLaunchArgument('use_lora', default_value='true',
                              description='Apply the task-specific LoRA weights.'),
        DeclareLaunchArgument('adapter', default_value='/workspace/models/pen_lora',
                              description='Adapter and normalization directory.'),
        RegisterEventHandler(OnProcessExit(
            target_action=recording,
            on_exit=[EmitEvent(event=Shutdown(reason='Recording process exited.'))],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=policy,
            on_exit=[EmitEvent(event=Shutdown(reason='Policy process exited.'))],
        )),
        recording,
        TimerAction(period=3.0, actions=[policy]),
    ])
