"""Check source syntax and configuration formats without loading model assets."""

import json
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    '.git', '.venv', '.pixi', '.cache', '.pytest_cache', '__pycache__',
    'outputs', 'build', 'install', 'log', 'cache', '.evaluation_deps',
}


def main():
    errors = []
    checked = 0
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts) or not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == '.py':
                compile(path.read_bytes(), str(relative), 'exec')
            elif suffix == '.json':
                json.loads(path.read_text(encoding='utf-8'))
            elif suffix in {'.yaml', '.yml'}:
                list(yaml.safe_load_all(path.read_text(encoding='utf-8')))
            elif suffix == '.toml' or path.name == 'uv.lock':
                tomllib.loads(path.read_text(encoding='utf-8'))
            elif suffix in {'.xml', '.xacro'}:
                ET.parse(path)
            else:
                continue
            checked += 1
        except (SyntaxError, ValueError, yaml.YAMLError, ET.ParseError) as error:
            errors.append(f'{relative}: {error}')
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(f'Source check failed: {len(errors)} issue(s).')
    print(f'Source check passed: {checked} files.')


if __name__ == '__main__':
    main()
