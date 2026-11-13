"""Extract the pinned RPent source for its native API planner."""
import argparse
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'ed6b9223fa8e9854964d067c94d7df3ded90e75a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as stage:
        with tarfile.open(args.assets / f'rpent-{REVISION}.tar.gz') as archive:
            archive.extractall(stage, filter='data')
        destination = runtime / 'RPent'
        if destination.exists():
            shutil.rmtree(destination)
        (Path(stage) / f'RPent-{REVISION}').replace(destination)
    print(f'Prepared RPent {REVISION}')


if __name__ == '__main__':
    main()
