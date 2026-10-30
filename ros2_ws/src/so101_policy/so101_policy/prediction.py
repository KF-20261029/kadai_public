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

"""Check one trained-policy prediction using live ROS observations."""

import rclpy
from so101_policy.model import load_policy
from so101_policy.observation import capture_initial_observation, ObservationReceiver
import torch


def main(args=None):
    """Load the policy and predict one action chunk without commanding the robot."""
    rclpy.init(args=args)
    node = ObservationReceiver()
    try:
        node.declare_parameter('base_model', '/cache/model')
        node.declare_parameter('adapter', '/workspace/models/pen_lora')
        policy = load_policy(
            node.get_parameter('base_model').value,
            node.get_parameter('adapter').value,
        )
        capture_initial_observation(node)
        batch = node.batch('cuda')
        print('Predicting action...', flush=True)
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        if tuple(actions.shape) != (1, 50, 6):
            raise RuntimeError('Expected one chunk of 50 actions for six joints.')
        if not torch.isfinite(actions).all():
            raise RuntimeError('The predicted actions contain non-finite values.')
        print('Prediction ready.', flush=True)
        print('Action shape:', tuple(actions.shape))
        print('All actions finite:', bool(torch.isfinite(actions).all().item()))
        print('First action (arm degrees, gripper percent):', actions[0, 0].float().cpu().tolist())
        print('Observation timestamp spread (seconds):', node.timestamp_spread(), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
