#!/usr/bin/env bash
set -eo pipefail

seed=${1:-7100}
output_root=${REALTIME_OUTPUT:-/workspace/outputs/realtime_comparison}
evaluator_path=${EVAL_PYTHONPATH:-/workspace/.evaluation_deps}
duration_seconds=${REALTIME_DURATION:-75.0}
lead_seconds=${REALTIME_LEAD_SECONDS:-0.35}
adapter_path=${POLICY_ADAPTER:-/workspace/models/pen_lora}
state_file="/workspace/config/initial_states/seed_${seed}.yaml"
prefix="$output_root/seed_${seed}"
mkdir -p "$output_root"
if [[ ! -f "$state_file" || -e "$prefix" || -e "$prefix.score.json" || -e "$prefix.run.log" ]]; then
  echo "Missing initial state or existing result: $prefix" >&2
  exit 1
fi

source /opt/mujoco337/control/setup.bash
source /workspace/ros2_ws/install/setup.bash
echo "Preparing initial state for seed $seed."
ros2 service call /mujoco_ros2_control_node/set_pause mujoco_ros2_control_msgs/srv/SetPause '{paused: true}' > "$prefix.reset.log"
grep -q 'success=True' "$prefix.reset.log"
ros2 service call /mujoco_ros2_control_node/reset_world mujoco_ros2_control_msgs/srv/ResetWorld "$(cat "$state_file")" >> "$prefix.reset.log"
grep -q 'Successfully reset' "$prefix.reset.log"
ros2 topic pub --once /trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{joint_names: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.7854]}]}" > "$prefix.initial_command.log"
echo 'Starting the continuously running simulation trial.'
ros2 launch /workspace/launch/realtime_trial.launch.py "output:=$prefix" "seed:=$seed" "duration_seconds:=$duration_seconds" "lead_seconds:=$lead_seconds" "adapter:=$adapter_path" | tee "$prefix.run.log"
grep -q 'Real-time trial complete.' "$prefix.run.log"
PYTHONPATH="$evaluator_path${PYTHONPATH:+:$PYTHONPATH}" python /workspace/scripts/score_realtime_bag.py "$prefix" --clock-window "$prefix/metrics.json" --expected-seconds "$duration_seconds" --output "$prefix.score.json"
echo "Completed real-time trial (seed $seed)."
