"""Compare paired paused and continuously running LoRA trials."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paused', type=Path, default=Path('/workspace/outputs/lora_comparison/lora'))
    parser.add_argument('--continuous', type=Path, default=Path('/workspace/outputs/realtime_comparison'))
    parser.add_argument('--seed-start', type=int, default=7100)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError('The number of trials must be positive.')
    summary_path = args.continuous / 'summary.json'
    if summary_path.exists():
        raise FileExistsError(summary_path)
    pairs, inference, observation_age, idle = [], [], [], []
    overlap_count = 0
    execution_conditions = set()
    for seed in range(args.seed_start, args.seed_start + args.trials):
        paused = json.loads((args.paused / f'seed_{seed}.score.json').read_text())
        continuous = json.loads((args.continuous / f'seed_{seed}.score.json').read_text())
        metrics = json.loads((args.continuous / f'seed_{seed}' / 'metrics.json').read_text())
        if metrics['seed'] != seed or not metrics['continuous_simulation'] or metrics['simulation_seconds'] < 75.0:
            raise ValueError(f'Invalid continuous-trial metrics for seed {seed}.')
        if not isinstance(paused.get('success'), bool) or not isinstance(continuous.get('success'), bool):
            raise ValueError(f'Invalid success decision for seed {seed}.')
        if not any(chunk['inference_sim_seconds'] > 0 for chunk in metrics['chunks']):
            raise ValueError(f'No simulation advance during inference for seed {seed}.')
        if metrics.get('executed_actions', 50) != 50 or metrics.get('compensate_latency', False):
            raise ValueError(f'Unexpected execution settings for seed {seed}.')
        execution_conditions.add(metrics['lead_seconds'])
        pairs.append({'seed': seed, 'paused': paused['success'], 'continuous': continuous['success']})
        for chunk in metrics['chunks']:
            inference.append(chunk['inference_wall_seconds'])
            observation_age.append(chunk['observation_age_at_send_seconds'])
            if chunk['idle_before_send_seconds'] is not None:
                idle.append(chunk['idle_before_send_seconds'])
            overlap_count += bool(chunk['inference_started_while_action_active'])
    if not inference or not overlap_count:
        raise ValueError('Missing inference or overlap measurements.')
    if len(execution_conditions) != 1:
        raise ValueError('The continuous trials use different execution settings.')
    lead_seconds = execution_conditions.pop()
    counts = {name: sum(pair[name] for pair in pairs) for name in ('paused', 'continuous')}
    result = {'trials_per_condition': args.trials, 'success_criterion':
              'Three pens inserted and released for at least 2 seconds',
              'paused_successes': counts['paused'], 'continuous_successes': counts['continuous'],
              'paused_success_rate': counts['paused'] / args.trials,
              'continuous_success_rate': counts['continuous'] / args.trials,
              'median_inference_wall_seconds': float(np.median(inference)),
              'predicted_actions': 50, 'executed_actions': 50,
              'command_rate_hz': 30.0, 'lead_seconds': lead_seconds,
              'median_observation_age_seconds': float(np.median(observation_age)),
              'max_idle_before_send_seconds': max(idle) if idle else None,
              'chunks_with_inference_during_prior_action': overlap_count, 'pairs': pairs}
    summary_path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Paused: {counts['paused']}/{args.trials} ({100 * result['paused_success_rate']:.0f}%)")
    print(f"Continuous: {counts['continuous']}/{args.trials} ({100 * result['continuous_success_rate']:.0f}%)")
    print(f"Median inference: {result['median_inference_wall_seconds']:.3f} s")
    print(f"Median observation age at command: {result['median_observation_age_seconds']:.3f} s")


if __name__ == '__main__':
    main()
