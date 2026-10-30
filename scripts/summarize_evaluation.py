"""Summarize paired evaluation results without counting incomplete trials."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--seed-start', type=int, default=7100)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError('The trial count must be positive.')
    output = args.directory / 'summary.json'
    if output.exists():
        raise FileExistsError(output)
    pairs = []
    for seed in range(args.seed_start, args.seed_start + args.trials):
        pair = {'seed': seed}
        for mode in ('base', 'lora'):
            path = args.directory / mode / f'seed_{seed}.score.json'
            score = json.loads(path.read_text())
            if not isinstance(score.get('success'), bool):
                raise ValueError(f'Invalid success result: {path}')
            pair[mode] = score['success']
        pairs.append(pair)
    counts = {mode: sum(pair[mode] for pair in pairs) for mode in ('base', 'lora')}
    result = {
        'trials_per_model': args.trials,
        'criterion': 'Three pens inserted and released for at least 2 seconds',
        'normalization': 'Task state and action statistics shared by both models',
        'base_successes': counts['base'],
        'lora_successes': counts['lora'],
        'base_success_rate': counts['base'] / args.trials,
        'lora_success_rate': counts['lora'] / args.trials,
        'pairs': pairs,
    }
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Base: {counts['base']}/{args.trials} ({100 * result['base_success_rate']:.0f}%)")
    print(f"LoRA: {counts['lora']}/{args.trials} ({100 * result['lora_success_rate']:.0f}%)")


if __name__ == '__main__':
    main()
