"""Prepare the tested OpenPI source snapshot and its Jetson runtime patches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'vendor/openpi'
    manifest = json.loads((source / 'SOURCE.sha256.json').read_text())
    for name, checksum in manifest.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'OpenPI source checksum mismatch: {name}')
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as stage:
        staged = Path(stage) / 'openpi-src'
        shutil.copytree(source / 'src', staged)
        shutil.copytree(source / 'packages/openpi-client/src/openpi_client', staged / 'openpi_client')
        subprocess.run(['patch', '--batch', '-p1', '-i', str(ROOT / 'inference.patch')], cwd=staged, check=True)
        destination = runtime / 'openpi-src'
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    replacement = destination / 'openpi/models_pytorch/transformers_replace'
    transformers = Path(sysconfig.get_path('purelib')) / 'transformers'
    # Replace files atomically to preserve uv's shared cache and other environments' hardlinks.
    for source_file in replacement.rglob('*'):
        if not source_file.is_file():
            continue
        target = transformers / source_file.relative_to(replacement)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.jetson.tmp')
        shutil.copyfile(source_file, temporary)
        temporary.replace(target)
    # This public NVIDIA archive is verified by fetch_assets before extraction.
    with tarfile.open(args.assets / 'libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz') as archive:
        for member in archive.getmembers():
            target = (runtime / member.name).resolve()
            if not target.is_relative_to(runtime.resolve()):
                raise ValueError(f'Archive path escapes runtime: {member.name}')
            if member.issym() and not (target.parent / member.linkname).resolve().is_relative_to(runtime.resolve()):
                raise ValueError(f'Archive link escapes runtime: {member.name}')
        archive.extractall(runtime)
    print('Prepared OpenPI source, Transformers replacements and cuDSS')


if __name__ == '__main__':
    main()
