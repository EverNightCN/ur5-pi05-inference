"""Validate a separately supplied checkpoint and select it for this checkout."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checkpoint import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    audit = validate_checkpoint(checkpoint)
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    link = runtime / 'checkpoint-current'
    temporary = runtime / f'.checkpoint-{uuid.uuid4().hex}'
    temporary.symlink_to(checkpoint, target_is_directory=True)
    temporary.replace(link)
    print(json.dumps(dict(checkpoint=str(checkpoint),
                          model_sha256=audit['files_sha256']['model.safetensors'])))


if __name__ == '__main__':
    main()
