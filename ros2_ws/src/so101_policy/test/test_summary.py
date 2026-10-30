"""Check paired success aggregation and rejection of incomplete results."""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / 'scripts/summarize_evaluation.py'


def write_scores(root):
    for mode, successful_seeds in [('base', set()), ('lora', {7100, 7102})]:
        directory = root / mode
        directory.mkdir()
        for seed in range(7100, 7110):
            (directory / f'seed_{seed}.score.json').write_text(
                json.dumps({'success': seed in successful_seeds}))


def summarize(root):
    return subprocess.run([sys.executable, str(SCRIPT), str(root)],
                          capture_output=True, text=True)


def test_paired_counts_and_overwrite_guard(tmp_path):
    write_scores(tmp_path)
    result = summarize(tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['trials_per_model'] == 10
    assert summary['base_successes'] == 0
    assert summary['lora_successes'] == 2
    assert len(summary['pairs']) == 10
    assert summarize(tmp_path).returncode != 0


def test_missing_trial_is_not_counted_as_failure(tmp_path):
    write_scores(tmp_path)
    (tmp_path / 'lora/seed_7109.score.json').unlink()
    assert summarize(tmp_path).returncode != 0
    assert not (tmp_path / 'summary.json').exists()


def test_nonboolean_success_is_rejected(tmp_path):
    write_scores(tmp_path)
    (tmp_path / 'lora/seed_7100.score.json').write_text('{"success": "false"}')
    assert summarize(tmp_path).returncode != 0
    assert not (tmp_path / 'summary.json').exists()


def write_continuous_metrics(root):
    directory = root / 'continuous'
    directory.mkdir()
    for seed in range(7100, 7110):
        (directory / f'seed_{seed}.score.json').write_text(
            json.dumps({'success': seed == 7109}))
        trial = directory / f'seed_{seed}'
        trial.mkdir()
        metrics = {
            'seed': seed, 'continuous_simulation': True, 'simulation_seconds': 75,
            'lead_seconds': 0.35,
            'chunks': [{
                'inference_sim_seconds': 0.14, 'inference_wall_seconds': 0.14,
                'observation_age_at_send_seconds': 0.38,
                'idle_before_send_seconds': 0.003,
                'inference_started_while_action_active': True,
            }],
        }
        (trial / 'metrics.json').write_text(json.dumps(metrics))
    return directory


def summarize_continuous(root, directory):
    return subprocess.run([
        sys.executable, str(ROOT / 'scripts/summarize_realtime.py'),
        '--paused', str(root / 'lora'), '--continuous', str(directory),
    ], capture_output=True, text=True)


def test_continuous_counts_and_legacy_metrics(tmp_path):
    write_scores(tmp_path)
    directory = write_continuous_metrics(tmp_path)
    result = summarize_continuous(tmp_path, directory)
    assert result.returncode == 0, result.stderr
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['paused_successes'] == 2
    assert summary['continuous_successes'] == 1
    assert summary['executed_actions'] == 50
    assert summary['median_inference_wall_seconds'] == 0.14


def test_other_experiment_settings_are_rejected(tmp_path):
    write_scores(tmp_path)
    directory = write_continuous_metrics(tmp_path)
    path = directory / 'seed_7100/metrics.json'
    metrics = json.loads(path.read_text())
    metrics['executed_actions'] = 10
    path.write_text(json.dumps(metrics))
    assert summarize_continuous(tmp_path, directory).returncode != 0
    assert not (directory / 'summary.json').exists()
