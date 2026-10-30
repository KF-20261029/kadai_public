# Copyright 2026 SO-101 task contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Receive synchronized ROS observations in the trained policy's conventions."""

import time

from message_filters import ApproximateTimeSynchronizer, Subscriber
import mujoco
from mujoco_ros2_control_msgs.srv import StepSimulation
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.task import Future
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
import torch


JOINT_NAMES = (
    'shoulder_pan', 'shoulder_lift', 'elbow_flex',
    'wrist_flex', 'wrist_roll', 'gripper',
)


def image_tensor(message, device):
    """Pack the driver's RGB8 image, respecting row padding, as a model tensor."""
    if message.encoding != 'rgb8' or message.width <= 0 or message.height <= 0:
        raise ValueError('A nonempty rgb8 image is required.')
    if message.step < message.width * 3 or len(message.data) != message.height * message.step:
        raise ValueError('Image row stride or byte count is inconsistent.')
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = rows[:, :message.width * 3].reshape(message.height, message.width, 3)
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32,
    ) / 255.0


class ObservationReceiver(Node):
    """Use standard ROS message filters to collect the two images and joint state."""

    def __init__(self, image_qos=None):
        super().__init__('so101_observation')
        if image_qos is None:
            image_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.declare_parameter('scene_path', '/workspace/simulation/scene.xml')
        self.declare_parameter('sync_slop', 0.02)
        self.declare_parameter('instruction', 'Grab pens and place into pen holder')
        model = mujoco.MjModel.from_xml_path(self.get_parameter('scene_path').value)
        self.ranges = np.array([
            [max(model.joint(name).range[0], model.actuator(name).ctrlrange[0]),
             min(model.joint(name).range[1], model.actuator(name).ctrlrange[1])]
            for name in JOINT_NAMES
        ])
        if not np.isfinite(self.ranges).all() or not np.all(self.ranges[:, 1] > self.ranges[:, 0]):
            raise ValueError('Joint and actuator ranges are invalid.')
        self.messages = None
        self.ready = Future()
        self.subscribers = [
            Subscriber(self, Image, '/mujoco_camera_plugin/top/color', image_qos),
            Subscriber(
                self, Image, '/mujoco_camera_plugin/wrist_cam/color', image_qos,
            ),
            Subscriber(self, JointState, '/joint_states', qos_profile_sensor_data),
        ]
        self.synchronizer = ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=self.get_parameter('sync_slop').value,
        )
        self.synchronizer.registerCallback(self.receive)

    def receive(self, top, wrist, joints):
        """Keep the latest set selected by the standard timestamp synchronizer."""
        self.messages = (top, wrist, joints)
        if not self.ready.done():
            self.ready.set_result(True)

    def state_values(self):
        """Return degrees for the arm and percent opening for the gripper."""
        if self.messages is None:
            raise RuntimeError('No synchronized observation has been received.')
        message = self.messages[2]
        if (len(message.name) != len(message.position) or
                len(set(message.name)) != len(message.name)):
            raise ValueError('Joint names and positions are inconsistent.')
        positions = dict(zip(message.name, message.position))
        if any(name not in positions for name in JOINT_NAMES):
            raise ValueError('The observation is missing a required joint.')
        radians = np.array([positions[name] for name in JOINT_NAMES], dtype=np.float64)
        if not np.isfinite(radians).all():
            raise ValueError('Joint positions contain non-finite values.')
        values = np.rad2deg(radians)
        lower, upper = self.ranges[5]
        values[5] = (radians[5] - lower) / (upper - lower) * 100.0
        return values.astype(np.float32)

    def batch(self, device='cpu'):
        """Build the two-camera input expected by the existing pi0.5 implementation."""
        state = self.state_values()
        top, wrist, _ = self.messages
        return {
            'task': [self.get_parameter('instruction').value],
            'observation.state': torch.from_numpy(state).unsqueeze(0).to(device),
            'observation.images.base_0_rgb': image_tensor(top, device),
            'observation.images.left_wrist_0_rgb': image_tensor(wrist, device),
        }

    def timestamp_spread(self):
        """Return the largest header timestamp difference in the received set."""
        if self.messages is None:
            raise RuntimeError('No synchronized observation has been received.')
        times = [message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                 for message in self.messages]
        return (max(times) - min(times)) * 1e-9


def capture_initial_observation(node):
    """Request fresh data with the paused simulator's standard stepping service."""
    if node.messages is not None:
        raise RuntimeError('Initial observation capture requires a new receiver.')
    client = node.create_client(StepSimulation, '/mujoco_ros2_control_node/step_simulation')
    if not client.wait_for_service(timeout_sec=10.0):
        raise RuntimeError('The simulator stepping service is unavailable.')
    deadline = time.monotonic() + 5.0
    while not all(item.sub.get_publisher_count() for item in node.subscribers):
        if time.monotonic() >= deadline:
            raise RuntimeError('Camera or joint-state publishers are unavailable.')
        rclpy.spin_once(node, timeout_sec=0.05)
    triggers = [node.create_client(Trigger, '/mujoco_camera_plugin/top/trigger'),
                node.create_client(Trigger, '/mujoco_camera_plugin/wrist_cam/trigger')]
    polled = all(trigger.wait_for_service(timeout_sec=1.0) for trigger in triggers)
    print('Capturing observations during 30 native simulation steps...', flush=True)
    future = client.call_async(StepSimulation.Request(steps=29 if polled else 30))
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done() or future.result() is None:
        raise RuntimeError('The simulation step request did not complete.')
    if not future.result().success:
        raise RuntimeError(future.result().message)
    if polled:
        for trigger in triggers:
            future = trigger.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
            if not future.done() or future.result() is None or not future.result().success:
                raise RuntimeError('A camera trigger request did not complete.')
        future = client.call_async(StepSimulation.Request(steps=1))
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('The final simulation step request did not complete.')
    rclpy.spin_until_future_complete(node, node.ready, timeout_sec=5.0)
    if not node.ready.done():
        raise RuntimeError('No synchronized camera and joint-state set arrived.')


def main(args=None):
    """Check input capture by advancing the paused driver for thirty steps."""
    rclpy.init(args=args)
    node = ObservationReceiver()
    try:
        capture_initial_observation(node)
        batch = node.batch()
        print('Observation ready.', flush=True)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(key + ':', tuple(value.shape))
        print('State (arm degrees, gripper percent):', batch['observation.state'][0].tolist())
        print('Timestamp spread (seconds):', node.timestamp_spread())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
