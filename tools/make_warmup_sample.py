"""Generate synthetic RGB/state inputs for service compilation and warmup."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / '.runtime/warmup.npz'
    output.parent.mkdir(exist_ok=True)
    state = np.zeros(13, dtype=np.float32)
    state[2:6] = [.4, 0, np.pi, 0]
    state[6:12] = np.deg2rad([-180, -90, -90, -90, 90, -90])
    state[12] = 1
    np.savez_compressed(output, **{
        'observation/image': np.zeros((720, 1280, 3), dtype=np.uint8),
        'observation/wrist_image': np.zeros((720, 1280, 3), dtype=np.uint8),
        'observation/state': state,
    })
    print(f'Created synthetic warmup input: {output}')


if __name__ == '__main__':
    main()
