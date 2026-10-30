"""Apply the pinned release video patch inside the isolated training environment."""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    '/opt/vlash': '22cbabfee0f57874987c75a35a7dac129e695db0',
    '/opt/so101-pi05-base': '73167f4dba88633aeec95ba7a3b3881916fa9aaa',
}


def main():
    for directory, revision in EXPECTED.items():
        actual = subprocess.check_output(['git', '-C', directory, 'rev-parse', 'HEAD'],
                                         text=True).strip()
        if actual != revision:
            raise RuntimeError('Unexpected source revision: ' + directory)
    package = Path(importlib.util.find_spec('lerobot').origin).parent
    source = Path('/opt/so101-pi05-base/training-docker/patched/video_utils.py')
    shutil.copyfile(source, package / 'datasets/video_utils.py')
    versions = {name: importlib.metadata.version(name) for name in
                ('vlash', 'lerobot', 'torch', 'torchcodec', 'transformers', 'peft', 'mujoco')}
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    (output / 'environment.json').write_text(json.dumps(versions, indent=2) + '\n')
    print('Pinned sources and release video patch prepared.')


if __name__ == '__main__':
    main()
