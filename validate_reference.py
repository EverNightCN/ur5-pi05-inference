"""Validate production graph inference against fixed-noise JAX checkpoint references."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from runtime import load_policy, read_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint, attention_backend='sdpa', compile_mode='default',
                         cuda_graph=True, compile_all=True, cache_padding=True)
    results = []
    reports = []
    for index in (0, 1, 0):
        observation = read_observation(args.references / f'parity-{index}-sample.npz')
        with np.load(args.references / f'parity-{index}-reference.npz', allow_pickle=False) as reference:
            noise = reference['noise']
            expected = reference['actions'][0]
        started = time.monotonic()
        actions = np.asarray(policy.infer(observation, noise=noise)['actions'])
        elapsed = time.monotonic() - started
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError('Invalid full policy output')
        difference = actions - expected
        rotations = Rotation.from_rotvec(actions[:, 3:6]).inv() * Rotation.from_rotvec(expected[:, 3:6])
        report = dict(sample=index, infer_s=elapsed,
            max_abs_per_dimension=np.abs(difference).max(0).tolist(),
            position_error_mm=np.percentile(np.linalg.norm(difference[:, :3], axis=1) * 1000, [50, 95, 100]).tolist(),
            rotation_error_deg=np.percentile(np.rad2deg(rotations.magnitude()), [50, 95, 100]).tolist(),
            max_gripper_error=float(np.abs(difference[:, 6]).max()))
        reports.append(report)
        results.append(actions)
        print(json.dumps(report), flush=True)
    np.testing.assert_array_equal(results[0], results[2])
    if np.array_equal(results[0], results[1]):
        raise AssertionError('Changed observations/noise produced identical actions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(metadata=policy.metadata, input_refresh=True,
                                         fixed_noise_repeat_exact=True, reports=reports), indent=2))
    np.savez_compressed(args.output.with_suffix('.npz'), actions=results)


if __name__ == '__main__':
    main()
