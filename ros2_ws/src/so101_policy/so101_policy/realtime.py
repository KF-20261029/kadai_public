"""Execute policy chunks while the ROS-controlled simulation keeps running."""

from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from mujoco_ros2_control_msgs.srv import SetPause
import numpy as np
import rclpy
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
import torch
from trajectory_msgs.msg import JointTrajectoryPoint

from so101_policy.control import PolicyReceiver, call, duration, trigger_cameras
from so101_policy.model import load_policy
from so101_policy.observation import JOINT_NAMES


def infer(policy, batch):
    started = time.monotonic()
    with torch.inference_mode():
        prediction = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    if tuple(prediction.shape) != (1, 50, 6) or not torch.isfinite(prediction).all():
        raise RuntimeError('Expected 50 finite actions for six joints.')
    return prediction[0].float().cpu().numpy().astype(np.float64), time.monotonic() - started


def spin_until(node, predicate, deadline_seconds=15.0):
    deadline = time.monotonic() + deadline_seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise RuntimeError('Timed out while waiting for ROS data or inference.')
        rclpy.spin_once(node, timeout_sec=0.02)


def capture(node, triggers):
    spin_until(node, lambda: node.latest_clock_ns is not None)
    minimum = node.latest_clock_ns
    trigger_cameras(node, triggers)
    stamp, age = node.wait_for_fresh_observation(minimum)
    return node.batch('cuda'), stamp, age


def send_chunk(node, action, raw):
    reference_ns = node.latest_clock_ns
    targets = np.empty_like(raw)
    targets[:, :5] = np.deg2rad(raw[:, :5])
    lower, upper = node.ranges[5]
    targets[:, 5] = lower + raw[:, 5] / 100.0 * (upper - lower)
    targets = np.clip(targets, node.ranges[:, 0], node.ranges[:, 1])
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(JOINT_NAMES)
    goal.trajectory.points.append(JointTrajectoryPoint(
        positions=targets[0].tolist(), time_from_start=duration(0)))
    for index, target in enumerate(targets):
        goal.trajectory.points.append(JointTrajectoryPoint(
            positions=target.tolist(), time_from_start=duration(index + 1)))
    accepted = action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, accepted, timeout_sec=10.0)
    if not accepted.done() or accepted.result() is None or not accepted.result().accepted:
        raise RuntimeError('The trajectory controller did not accept the goal.')
    handle = accepted.result()
    execution_ns = round(50 / 30 * 1e9)
    return handle, handle.get_result_async(), {
        'trajectory_reference_ns': reference_ns,
        'trajectory_duration_ns': execution_ns,
        'planned_end_ns': reference_ns + execution_ns,
        'first_target_after_reference_ns': round(1e9 / 30),
    }


def main(args=None):
    rclpy.init(args=args)
    node = PolicyReceiver()
    action = None
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        node.declare_parameter('seed', 7100)
        node.declare_parameter('base_model', '/cache/model')
        node.declare_parameter('adapter', '/workspace/models/pen_lora')
        node.declare_parameter('duration_seconds', 75.0)
        node.declare_parameter('lead_seconds', 0.35)
        node.declare_parameter('metrics_path', '/workspace/outputs/realtime_trial/metrics.json')
        seed = node.get_parameter('seed').value
        duration_s = node.get_parameter('duration_seconds').value
        lead = node.get_parameter('lead_seconds').value
        execution_seconds = 50 / 30
        if seed < 0 or duration_s <= 0 or not 0.0 <= lead <= execution_seconds:
            raise ValueError('Invalid seed, duration, or inference lead.')
        metrics_path = Path(node.get_parameter('metrics_path').value)
        if metrics_path.exists():
            raise FileExistsError(metrics_path)
        pause = node.create_client(SetPause, '/mujoco_ros2_control_node/set_pause')
        triggers = [node.create_client(Trigger, '/mujoco_camera_plugin/top/trigger'),
                    node.create_client(Trigger, '/mujoco_camera_plugin/wrist_cam/trigger')]
        action = ActionClient(node, FollowJointTrajectory, '/trajectory_controller/follow_joint_trajectory')
        if not action.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('The trajectory controller is unavailable.')
        policy = load_policy(node.get_parameter('base_model').value,
                             node.get_parameter('adapter').value, use_lora=True)
        torch.manual_seed(seed)
        spin_until(node, lambda: all(item.sub.get_publisher_count() for item in node.subscribers))
        call(node, pause, SetPause.Request(paused=False))
        spin_until(node, lambda: node.latest_clock_ns is not None)
        start_ns = node.latest_clock_ns
        wall_start = time.monotonic()
        end_ns = start_ns + round(duration_s * 1e9)
        pending = None
        active_handle = None
        active_result = None
        active_start = None
        active_end_ns = None
        active_completed_ns = None
        pending_stamp = None
        inference_start_ns = None
        inference_end_ns = None
        rows = []
        while node.latest_clock_ns < end_ns:
            if active_result is not None and active_result.done() and active_completed_ns is None:
                active_completed_ns = node.latest_clock_ns
                result = active_result.result()
                if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
                    raise RuntimeError('The preceding trajectory did not succeed.')
            if pending is None and (active_result is None or active_result.done() or
                                    node.latest_clock_ns >= active_end_ns - round(lead * 1e9)):
                if active_result is not None and end_ns - node.latest_clock_ns < 500_000_000:
                    spin_until(node, lambda: node.latest_clock_ns >= end_ns, deadline_seconds=15.0)
                    break
                batch, pending_stamp, capture_age = capture(node, triggers)
                inference_start_ns = node.latest_clock_ns
                inference_end_ns = None
                pending = pool.submit(infer, policy, batch)
                was_active = active_result is not None and not active_result.done()
                print(f'Observation captured: age {capture_age:.3f}s, previous action active: {was_active}.', flush=True)
            if pending is not None and pending.done() and inference_end_ns is None:
                inference_end_ns = node.latest_clock_ns
            if pending is not None and pending.done() and (active_result is None or active_result.done()):
                raw, inference_wall = pending.result()
                if node.latest_clock_ns >= end_ns:
                    break
                new_handle, new_result, timing = send_chunk(node, action, raw)
                active_start = node.latest_clock_ns
                active_end_ns = active_start + round(execution_seconds * 1e9)
                age = (active_start - pending_stamp) * 1e-9
                rows.append({'chunk': len(rows) + 1, 'inference_wall_seconds': inference_wall,
                             'predicted_actions': len(raw), 'executed_actions': 50,
                             **timing,
                             'inference_sim_seconds': (inference_end_ns - inference_start_ns) * 1e-9,
                             'observation_age_at_send_seconds': age,
                             'inference_started_while_action_active': was_active,
                             'idle_before_send_seconds': None if active_completed_ns is None else
                             max(0, active_start - active_completed_ns) * 1e-9,
                             'sim_time_at_send_seconds': (active_start - start_ns) * 1e-9})
                print(f'Chunk {len(rows)} sent: inference {inference_wall:.3f}s, '
                      f'observation age {age:.3f}s.',
                      flush=True)
                active_result = new_result
                active_handle = new_handle
                active_completed_ns = None
                pending = None
            rclpy.spin_once(node, timeout_sec=0.02)
        if active_handle is not None and active_result is not None and not active_result.done():
            cancelled = active_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancelled, timeout_sec=5.0)
            if not cancelled.done() or cancelled.result() is None:
                raise RuntimeError('The final trajectory cancellation did not complete.')
        wall_elapsed = time.monotonic() - wall_start
        result = {'seed': seed, 'start_clock_ns': start_ns, 'end_clock_ns': end_ns,
                  'simulation_seconds': (node.latest_clock_ns - start_ns) * 1e-9,
                  'wall_seconds': wall_elapsed, 'chunks_sent': len(rows), 'chunks': rows,
                  'continuous_simulation': True, 'lead_seconds': lead,
                  'predicted_actions': 50, 'executed_actions': 50,
                  'command_rate_hz': 30.0, 'execution_seconds': execution_seconds}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2) + '\n')
        print('Real-time trial complete.', flush=True)
    finally:
        pool.shutdown(wait=True)
        if action is not None:
            action.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
