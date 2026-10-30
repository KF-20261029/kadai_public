"""Extract reported training metrics without copying machine-specific log text."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


PATTERN = re.compile(
    r'\bstep:(\S+).*?\bloss:([\d.eE+-]+)\s+grdn:([\d.eE+-]+)'
    r'\s+lr:([\d.eE+-]+)\s+updt_s:([\d.eE+-]+)\s+data_s:([\d.eE+-]+)'
)
FIELDS = ['record', 'logged_step', 'loss', 'gradient_norm', 'learning_rate',
          'update_seconds', 'data_seconds']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary_path = args.output.with_suffix('.json')
    if args.output.exists() or summary_path.exists():
        raise FileExistsError('Choose new output paths.')
    data = args.log.read_bytes()
    rows = []
    for line in data.decode('utf-8', errors='replace').splitlines():
        match = PATTERN.search(line)
        if match:
            rows.append(dict(zip(FIELDS, [len(rows) + 1, match[1],
                                         *map(float, match.groups()[1:])])))
    if not rows:
        raise ValueError('No supported training metric records were found.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'source_log_sha256': hashlib.sha256(data).hexdigest(),
        'metric_records': len(rows),
        'first_reported_loss': rows[0]['loss'],
        'last_reported_loss': rows[-1]['loss'],
        'minimum_reported_loss': min(row['loss'] for row in rows),
        'step_format': 'Original log labels; K suffix values are rounded.',
        'metric_precision': 'Values retain the precision printed in the log.',
        'loss_scope': 'Training loss; not independent validation loss or task success.',
    }
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Extracted {len(rows)} training metric records.')


if __name__ == '__main__':
    main()
