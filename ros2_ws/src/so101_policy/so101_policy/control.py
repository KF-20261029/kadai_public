"""Run trained-policy action chunks through the ROS trajectory controller."""

import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from mujoco_ros2_control_msgs.srv import SetPause, StepSimulation
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from so101_policy.model import load_policy
from so101_policy.observation import JOINT_NAMES, ObservationReceiver
from so101_policy.timing import command_time_ns
from std_srvs.srv import Trigger
import torch
from trajectory_msgs.msg import JointTrajectoryPoint


def call(node, client, request, timeout=15.0):
    """Call a simulator service and require a successful response."""
    if not client.wait_for_service(timeout_sec=10.0):
        raise RuntimeError('The simulator service is unavailable.')
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        raise RuntimeError('The simulator service did not respond.')
    if not future.result().success:
        raise RuntimeError(future.result().message)
    return future.result()


def duration(index):
    """Represent one-thirtieth-second multiples as ROS durations."""
    nanoseconds = command_time_ns(index)
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class PolicyReceiver(ObservationReceiver):
    """Track simulation time alongside synchronized ROS observations."""

    def __init__(self):
        super().__init__(image_qos=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.latest_clock_ns = None
        self.clock_subscription = self.create_subscription(Clock, '/clock', self.receive_clock, qos_profile_sensor_data)

    def receive_clock(self, message):
        value = stamp_ns(message.clock)
        if self.latest_clock_ns is None or value > self.latest_clock_ns:
            self.latest_clock_ns = value

    def wait_for_fresh_observation(self, minimum_ns=0):
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.025)
            if self.messages is None or self.latest_clock_ns is None:
                continue
            stamps = [stamp_ns(message.header.stamp) for message in self.messages]
            age = (self.latest_clock_ns - min(stamps)) * 1e-9
            if min(stamps) >= minimum_ns and -0.03 <= age <= 0.10:
                return min(stamps), age
        raise RuntimeError('A recent synchronized observation was not received.')


def trigger_cameras(node, triggers):
    for client in triggers:
        call(node, client, Trigger.Request())


def capture_initial(node, step, triggers):
    deadline = time.monotonic() + 10.0
    while not all(item.sub.get_publisher_count() for item in node.subscribers):
        if time.monotonic() >= deadline:
            raise RuntimeError('Camera or joint-state publishers are unavailable.')
        rclpy.spin_once(node, timeout_sec=0.05)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    call(node, step, StepSimulation.Request(steps=29))
    trigger_cameras(node, triggers)
    call(node, step, StepSimulation.Request(steps=1))
    rclpy.spin_until_future_complete(node, node.ready, timeout_sec=5.0)
    if not node.ready.done():
        raise RuntimeError('No synchronized camera and joint-state set arrived.')


def execute(node, action, step, targets, triggers):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(JOINT_NAMES)
    goal.trajectory.points.append(JointTrajectoryPoint(positions=targets[0].tolist(), time_from_start=duration(0)))
    for index, target in enumerate(targets):
        goal.trajectory.points.append(JointTrajectoryPoint(positions=target.tolist(), time_from_start=duration(index + 1)))
    accepted = action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, accepted, timeout_sec=10.0)
    if not accepted.done() or accepted.result() is None or not accepted.result().accepted:
        raise RuntimeError('The trajectory controller did not accept the goal.')
    result = accepted.result().get_result_async()
    call(node, step, StepSimulation.Request(steps=499), timeout=30.0)
    trigger_cameras(node, triggers)
    call(node, step, StepSimulation.Request(steps=1))
    completion_steps = 0
    for attempt in range(4):
        deadline = time.monotonic() + 0.5
        while not result.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if result.done() or attempt == 3:
            break
        call(node, step, StepSimulation.Request(steps=1), timeout=10.0)
        completion_steps += 1
    if not result.done() or result.result() is None:
        raise RuntimeError('The trajectory result was not received while paused.')
    if result.result().status != GoalStatus.STATUS_SUCCEEDED:
        raise RuntimeError(f'Trajectory failed: {result.result().result.error_string}')
    return completion_steps


def main(args=None):
    """Predict and execute policy chunks, leaving the simulation paused."""
    rclpy.init(args=args)
    initializer = rclpy.create_node('so101_control_initializer')
    node = None
    action = None
    try:
        pause = initializer.create_client(SetPause, '/mujoco_ros2_control_node/set_pause')
        call(initializer, pause, SetPause.Request(paused=True))
        initializer.destroy_node()
        initializer = None
        node = PolicyReceiver()
        node.declare_parameter('base_model', '/cache/model')
        node.declare_parameter('adapter', '/workspace/models/pen_lora')
        node.declare_parameter('chunks', 1)
        node.declare_parameter('seed', -1)
        node.declare_parameter('use_lora', True)
        chunks = node.get_parameter('chunks').value
        seed = node.get_parameter('seed').value
        use_lora = node.get_parameter('use_lora').value
        if not isinstance(chunks, int) or not 1 <= chunks <= 100:
            raise ValueError('The number of chunks must be between 1 and 100.')
        if not isinstance(seed, int) or seed < -1:
            raise ValueError('The seed must be a nonnegative integer or -1.')
        step = node.create_client(StepSimulation, '/mujoco_ros2_control_node/step_simulation')
        triggers = [node.create_client(Trigger, '/mujoco_camera_plugin/top/trigger'),
                    node.create_client(Trigger, '/mujoco_camera_plugin/wrist_cam/trigger')]
        action = ActionClient(node, FollowJointTrajectory, '/trajectory_controller/follow_joint_trajectory')
        if not action.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('The trajectory controller action is unavailable.')
        policy = load_policy(
            node.get_parameter('base_model').value,
            node.get_parameter('adapter').value,
            use_lora=use_lora,
        )
        if seed >= 0:
            torch.manual_seed(seed)
            print(f'Inference seed: {seed}', flush=True)
        capture_initial(node, step, triggers)
        minimum_stamp = 0
        for chunk in range(chunks):
            observation_stamp, age = node.wait_for_fresh_observation(minimum_stamp)
            print(f'Chunk {chunk + 1}/{chunks}: observation age {age:.3f} seconds.', flush=True)
            batch = node.batch('cuda')
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            if tuple(prediction.shape) != (1, 50, 6) or not torch.isfinite(prediction).all():
                raise RuntimeError('Expected 50 finite actions for six joints.')
            raw = prediction[0].float().cpu().numpy().astype(np.float64)
            targets = np.empty_like(raw)
            targets[:, :5] = np.deg2rad(raw[:, :5])
            lower, upper = node.ranges[5]
            targets[:, 5] = lower + raw[:, 5] / 100.0 * (upper - lower)
            targets = np.clip(targets, node.ranges[:, 0], node.ranges[:, 1])
            print('Executing 50 actions during 500 simulation steps...', flush=True)
            completion_steps = execute(node, action, step, targets, triggers)
            print('Action chunk complete.', flush=True)
            print('Additional completion steps:', completion_steps, flush=True)
            minimum_stamp = observation_stamp + 1_600_000_000
        print('Simulation remains paused.', flush=True)
    finally:
        if action is not None:
            action.destroy()
        if node is not None:
            node.destroy_node()
        if initializer is not None:
            initializer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
