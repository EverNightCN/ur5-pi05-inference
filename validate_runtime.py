"""Check graph input refresh and record SDPA error against full eager inference."""
import argparse
import json
from pathlib import Path

from runtime import load_policy, read_observation, set_attention_backend, compile_components, cache_padding_image
from cuda_graph import CudaGraphSampler
import numpy as np
import torch


def error(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs_per_dimension": np.abs(delta).max(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint)
    observations = [read_observation(path) for path in args.samples]
    rng = np.random.default_rng(42)
    noises = [rng.standard_normal((50, 32)).astype(np.float32) for _ in range(2)]
    # A -> B -> A checks changing pixels, state, masks/tokens, and fresh noise.
    sequence = [(0, 0), (1, 1), (0, 0)]
    reference = []
    reports = []
    arrays = {}
    variants = [("eager", False, False, False), ("eager", True, False, False),
                ("eager", True, False, True), ("sdpa", False, False, False), ("sdpa", True, False, False)]
    if args.compile_all:
        variants.append(("sdpa", True, True, True))
    for attention, graph, compiled, cached in variants:
        if hasattr(policy._model, "cached_padding_image"):
            del policy._model.cached_padding_image
        set_attention_backend(policy._model, attention)
        if compiled:
            compile_components(policy._model, "default", all_components=True)
        if cached:
            cache_padding_image(policy._model)
        sampler = CudaGraphSampler(policy._model) if graph else policy._model.sample_actions
        captured = {}
        def record(*a, **kw):
            result = sampler(*a, **kw)
            captured["raw"] = result.detach().clone()
            return result
        policy._sample_actions = record
        variant = f"{attention}-{'graph' if graph else 'eager'}"
        if compiled:
            variant += "-compiled"
        if cached:
            variant += "-cached"
        current = []
        for index, (obs_index, noise_index) in enumerate(sequence):
            result = policy.infer(observations[obs_index], noise=noises[noise_index])["actions"]
            raw = captured["raw"][0].cpu().numpy()
            if raw.shape != (50, 32) or not np.isfinite(raw).all():
                raise ValueError("Invalid full model actions")
            current.append((raw, result))
            arrays[f"{variant}_{index}_raw"] = raw
            arrays[f"{variant}_{index}_actions"] = result
            if reference:
                report = {"variant": variant, "sequence_index": index,
                          "normalized": error(raw, reference[index][0]),
                          "physical": error(result, reference[index][1])}
                reports.append(report)
                print(json.dumps(report), flush=True)
                if attention == "eager":
                    np.testing.assert_array_equal(raw, reference[index][0])
                    np.testing.assert_array_equal(result, reference[index][1])
        if not reference:
            reference = current
        np.testing.assert_array_equal(current[0][0], current[2][0])
        if np.array_equal(current[0][0], current[1][0]):
            raise AssertionError("Changed observations and noise produced identical actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"eager_graph_exact": True, "input_refresh": True,
                                      "comparisons": reports}, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
