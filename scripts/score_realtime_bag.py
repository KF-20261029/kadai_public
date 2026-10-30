"""Score a ROS recording against the existing pen-placement criterion."""

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from mujoco_ros2_control_msgs.msg import FreeJointStateArray

from score_state import JOINT_NAMES, PEN_NAMES, StateJudge


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('--scene', default='/workspace/simulation/scene.xml')
    parser.add_argument('--expected-seconds', type=float, default=75.0)
    parser.add_argument('--clock-window', type=Path)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    clock_window = None
    if args.clock_window is not None:
        metrics = json.loads(args.clock_window.read_text())
        clock_window = (int(metrics['start_clock_ns']), int(metrics['end_clock_ns']))
        if clock_window[1] <= clock_window[0]:
            raise ValueError('The simulation clock window is invalid.')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    free_topic = '/free_joint_state_publisher/free_joint_states'
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/joint_states', free_topic]))
    joint_samples, pen_samples = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == '/joint_states':
            msg = deserialize_message(data, JointState)
            if clock_window is not None and not clock_window[0] <= stamp_ns(msg.header) <= clock_window[1]:
                continue
            positions = dict(zip(msg.name, msg.position))
            if all(name in positions for name in JOINT_NAMES):
                joint_samples.append((stamp_ns(msg.header), [positions[name] for name in JOINT_NAMES]))
        else:
            msg = deserialize_message(data, FreeJointStateArray)
            if clock_window is not None and not clock_window[0] <= stamp_ns(msg.header) <= clock_window[1]:
                continue
            free = {item.name: item.pose.pose for item in msg.free_joints}
            if all(name in free for name in PEN_NAMES):
                poses = []
                for name in PEN_NAMES:
                    pose = free[name]
                    poses.append([pose.position.x, pose.position.y, pose.position.z,
                                  pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
                pen_samples.append((stamp_ns(msg.header), poses))
    joint_samples.sort(key=lambda x: x[0])
    pen_samples.sort(key=lambda x: x[0])
    if len(joint_samples) < 2 or len(pen_samples) < 2:
        raise RuntimeError('Missing recorded joint or pen states.')
    times = np.array([sample[0] for sample in joint_samples], dtype=np.int64)
    judge = StateJudge(args.scene)
    last_time = None
    first_time = None
    active_start = None
    longest_ns = 0
    success = False
    matched = 0
    unmatched = 0
    max_gap_ns = 0
    final_placed = final_released = None
    for timestamp, poses in pen_samples:
        index = int(np.searchsorted(times, timestamp))
        candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
        nearest = min(candidates, key=lambda i: abs(int(times[i]) - timestamp))
        if abs(int(times[nearest]) - timestamp) > 4_000_000:
            unmatched += 1
            continue
        placed, released = judge.evaluate(joint_samples[nearest][1], poses)
        if first_time is None:
            first_time = timestamp
        if last_time is not None:
            max_gap_ns = max(max_gap_ns, timestamp - last_time)
        if released.all():
            if active_start is None or last_time is None or timestamp - last_time > 50_000_000:
                active_start = timestamp
            longest_ns = max(longest_ns, timestamp - active_start)
            success |= timestamp - active_start >= 2_000_000_000
        else:
            active_start = None
        last_time = timestamp
        matched += 1
        final_placed = placed.tolist()
        final_released = released.tolist()
    duration = (last_time - first_time) / 1e9 if first_time is not None and last_time is not None else 0.0
    if matched < 2 or duration < args.expected_seconds - 0.5 or max_gap_ns > 50_000_000:
        raise RuntimeError('Incomplete or insufficiently sampled state recording; no success decision made.')
    result = {'success': bool(success), 'criterion': 'Three pens inserted and released for at least 2 seconds',
              'final_placed': final_placed, 'final_released': final_released,
              'longest_seconds': longest_ns / 1e9, 'observed_seconds': duration,
              'matched_states': matched, 'unmatched_states': unmatched, 'largest_gap_seconds': max_gap_ns / 1e9}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
