"""Fetch public software dependencies and verify their pinned SHA-256 values."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['inference', 'harness'], required=True)
    parser.add_argument('--directory', type=Path, default=ROOT / '.assets')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for asset in json.loads((ROOT / 'requirements/assets.json').read_text()):
        if asset['group'] != args.group:
            continue
        target = args.directory / asset['name']
        if target.exists():
            if digest(target) != asset['sha256']:
                raise ValueError(f'Asset checksum mismatch: {target}')
            print(f'Verified {target.name}', flush=True)
            continue
        partial = target.with_name(target.name + '.part')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--connect-timeout', '20', '--output', str(partial), asset['url']], check=True)
        if digest(partial) != asset['sha256']:
            partial.unlink()
            raise ValueError(f'Download checksum mismatch: {target.name}')
        partial.replace(target)
        print(f'Downloaded and verified {target.name}', flush=True)


if __name__ == '__main__':
    main()
