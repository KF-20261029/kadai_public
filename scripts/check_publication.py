"""Check the public copy and optionally its staged Git blobs before publication."""

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
MODEL = 'models/pen_lora/lora_adapters/adapter_model.safetensors'
LIMIT = 100 * 1024 * 1024
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.evaluation_deps', '.venv',
            'outputs', 'build', 'install', 'log', '.cache', 'cache', '.pixi'}
PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'Hugging Face token': re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
    'API token': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}\b'),
    'host user path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/(?:share|Desktop|Documents|Downloads)/'),
    'cluster hostname': re.compile(r'\bytc\d{4}-\d+\b'),
}


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def scan_text(relative, data, errors):
    if b'\x00' in data:
        return
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError:
        return
    for number, line in enumerate(content.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            if pattern.search(line):
                errors.append(f'{relative}:{number}: possible {label}')
        for url in re.findall(r'https?://[^\s<>"\']+', line):
            try:
                parsed = urlsplit(url)
                if parsed.username is not None or parsed.password is not None:
                    errors.append(f'{relative}:{number}: credential-bearing URL')
                try:
                    address = ipaddress.ip_address(parsed.hostname or '')
                except ValueError:
                    continue
                if address.is_private and not address.is_loopback:
                    errors.append(f'{relative}:{number}: private-network URL')
            except ValueError:
                errors.append(f'{relative}:{number}: malformed URL')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    errors = []
    checked = 0
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_symlink():
            errors.append(f'{relative}: unexpected symlink')
            continue
        if not path.is_file():
            continue
        checked += 1
        if path.name in ('AGENTS.md', 'aaa', 'teacher_preview.json', '.env'):
            errors.append(f'{relative}: local-only file')
        if path.stat().st_size > LIMIT and str(relative) != MODEL:
            errors.append(f'{relative}: exceeds the ordinary GitHub file limit')
        if path.stat().st_size <= 10 * 1024 * 1024:
            scan_text(str(relative), path.read_bytes(), errors)
    checksums = json.loads((ROOT / 'models/pen_lora/CHECKSUMS.json').read_text())
    for relative, expected in checksums.items():
        path = ROOT / 'models/pen_lora' / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            errors.append(f'models/pen_lora/{relative}: model integrity check failed')
    attributes = (ROOT / '.gitattributes').read_text()
    if MODEL not in attributes or 'filter=lfs' not in attributes:
        errors.append('The adapter must be configured for Git LFS.')
    if args.staged:
        if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != ROOT:
            raise RuntimeError('Run the staged check in the public repository itself.')
        names = git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').decode().split('\x00')
        for name in filter(None, names):
            size = int(git('cat-file', '-s', ':' + name))
            if size > LIMIT:
                errors.append(f'{name}: raw large file staged; install Git LFS and restage it')
                continue
            data = git('show', ':' + name)
            scan_text(name, data, errors)
            if name == MODEL and not data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                errors.append(f'{name}: staged data is not a Git LFS pointer')
        for key in ('user.name', 'user.email'):
            setting = subprocess.run(['git', 'config', '--local', '--get', key],
                                     cwd=ROOT, capture_output=True, text=True).stdout.strip()
            if not setting or setting.startswith(('PUBLIC_', 'ACCOUNT_')):
                errors.append(f'{key}: set a deliberate repository-local public identity')
            elif key == 'user.email' and not setting.endswith('@users.noreply.github.com'):
                errors.append('user.email: use a repository-local GitHub noreply address')
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(f'Publication check failed: {len(errors)} issue(s).')
    print(f'Publication check passed: {checked} files; model checksums match.')
    if not args.staged:
        print('Run again with --staged after Git LFS and the local author identity are configured.')


if __name__ == '__main__':
    main()
