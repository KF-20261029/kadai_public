#!/usr/bin/env bash
set -eo pipefail
cd /workspace/training
pixi run python scripts/setup.py
python_config=/workspace/training/configs/pen_lora.json
pixi run python - "$python_config" <<'PY'
import json
from pathlib import Path
import sys
config=json.loads(Path(sys.argv[1]).read_text())
dataset=Path(config['dataset']['root'])
metadata=json.loads((dataset/'meta/info.json').read_text())
if metadata['codebase_version']!='v2.1' or metadata['total_episodes']!=55 or metadata['total_frames']!=99000:
    raise RuntimeError('This historical training configuration requires 55 complete v2.1 episodes / 99,000 frames.')
if Path(config['output_dir']).exists():
    raise FileExistsError('Choose a new output directory; training results are never overwritten or auto-resumed by this wrapper.')
if not Path('/cache/model/model.safetensors').is_file():
    raise FileNotFoundError('Prepare the pinned base checkpoint with the root model-download command first.')
print('Historical training input validated. Starting the upstream VLASH training CLI.')
PY
exec pixi run python /opt/vlash/vlash/train.py --config_path="$python_config"
