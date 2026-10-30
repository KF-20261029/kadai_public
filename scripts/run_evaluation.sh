#!/usr/bin/env bash
set -eo pipefail

mode=${1:?Specify base or lora}
case "$mode" in
  base) use_lora=false ;;
  lora) use_lora=true ;;
  *) echo 'Specify base or lora.' >&2; exit 2 ;;
esac

seed_start=${EVAL_SEED_START:-7100}
trial_count=${EVAL_TRIAL_COUNT:-10}
chunks=${EVAL_CHUNKS:-45}
output_root=${EVAL_OUTPUT:-/workspace/outputs/lora_comparison}
evaluator_path=${EVAL_PYTHONPATH:-/workspace/.evaluation_deps}
adapter_path=${POLICY_ADAPTER:-/workspace/models/pen_lora}
output_dir="$output_root/$mode"
mkdir -p "$output_dir"
source /opt/mujoco337/control/setup.bash
source /workspace/ros2_ws/install/setup.bash

for ((index = 0; index < trial_count; index++)); do
  seed=$((seed_start + index))
  state_file="/workspace/config/initial_states/seed_${seed}.yaml"
  prefix="$output_dir/seed_${seed}"
  if [[ ! -f "$state_file" || -e "$prefix" || -e "$prefix.score.json" || -e "$prefix.run.log" ]]; then
    echo "Missing initial state or existing result: $prefix" >&2
    exit 1
  fi
done

for ((index = 0; index < trial_count; index++)); do
  seed=$((seed_start + index))
  state_file="/workspace/config/initial_states/seed_${seed}.yaml"
  prefix="$output_dir/seed_${seed}"
  echo "Starting $mode trial $((index + 1))/$trial_count (seed $seed)."
  ros2 service call /mujoco_ros2_control_node/set_pause mujoco_ros2_control_msgs/srv/SetPause '{paused: true}' > "$prefix.reset.log"
  grep -q 'success=True' "$prefix.reset.log"
  ros2 service call /mujoco_ros2_control_node/reset_world mujoco_ros2_control_msgs/srv/ResetWorld "$(cat "$state_file")" >> "$prefix.reset.log"
  grep -q 'Successfully reset' "$prefix.reset.log"
  ros2 topic pub --once /trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{joint_names: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.7854]}]}" > "$prefix.initial_command.log"
  ros2 launch /workspace/launch/recorded_trial.launch.py "output:=$prefix" "chunks:=$chunks" "seed:=$seed" "use_lora:=$use_lora" "adapter:=$adapter_path" > "$prefix.run.log" 2>&1
  grep -q 'Simulation remains paused.' "$prefix.run.log"
  expected_seconds=$(python -c "print($chunks * 50 / 30)")
  PYTHONPATH="$evaluator_path${PYTHONPATH:+:$PYTHONPATH}" python /workspace/scripts/score_bag.py "$prefix" --expected-seconds "$expected_seconds" --output "$prefix.score.json"
  echo "Completed $mode trial $((index + 1))/$trial_count (seed $seed)."
done
