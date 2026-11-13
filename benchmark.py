"""Synchronised end-to-end benchmarks and fixed-noise regression artifacts."""
import argparse
import json
import os
import time
from pathlib import Path

from runtime import load_policy, read_observation
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--compile-all", action="store_true")
    parser.add_argument("--cache-padding", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--python-profile", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    obs = read_observation(args.sample)
    noise = np.random.default_rng(42).standard_normal((50, 32)).astype(np.float32)
    start = time.perf_counter()
    policy = load_policy(args.checkpoint, compile_mode=args.compile_mode, attention_backend=args.attention,
                         cuda_graph=args.cuda_graph, compile_all=args.compile_all,
                         cache_padding=args.cache_padding)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    print(json.dumps({"event": "loaded", "seconds": load_s}), flush=True)
    sample_actions = policy._sample_actions
    gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    def timed_sample(*a, **kw):
        gpu_start.record()
        result = sample_actions(*a, **kw)
        gpu_end.record()
        return result
    policy._sample_actions = timed_sample
    latencies = []
    gpu_spans = []
    outputs = []
    for i in range(args.iterations + 3):
        if i == 3:
            measurement_started = time.time()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = policy.infer(obs, noise=noise)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        actions = result["actions"]
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid action output")
        if i >= 3:
            latencies.append(elapsed)
            gpu_spans.append(gpu_start.elapsed_time(gpu_end))
            outputs.append(actions.copy())
        print(json.dumps({"iteration": i, "seconds": elapsed,
                          "model_gpu_span_ms": gpu_start.elapsed_time(gpu_end)}), flush=True)
    measurement_ended = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), actions=np.stack(outputs), noise=noise)
    report = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "compile_mode": args.compile_mode,
        "attention": args.attention,
        "cuda_graph": args.cuda_graph,
        "compile_all": args.compile_all,
        "cache_padding": args.cache_padding,
        "emulate_precision_casts": os.environ.get("TORCHINDUCTOR_EMULATE_PRECISION_CASTS") == "1",
        "checkpoint": str(args.checkpoint), "sample": str(args.sample),
        "action_horizon": 50, "denoise_steps": 10, "load_seconds": load_s,
        "iterations": args.iterations, "latencies_seconds": latencies,
        "measurement_started_unix": measurement_started,
        "measurement_ended_unix": measurement_ended,
        "model_gpu_span_ms": gpu_spans,
        "median_seconds": float(np.median(latencies)),
        "p95_seconds": float(np.percentile(latencies, 95)),
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "finite": True,
        "fixed_noise_max_repeat_difference": float(np.max(np.abs(np.stack(outputs) - outputs[0]))),
        "action_min": outputs[0].min(axis=0).tolist(),
        "action_max": outputs[0].max(axis=0).tolist(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if args.python_profile:
        import cProfile
        import io
        import pstats
        profiler = cProfile.Profile()
        profiler.runcall(policy.infer, obs, noise=noise)
        text = io.StringIO()
        pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(35)
        args.output.with_suffix(".python-profile.txt").write_text(text.getvalue())
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            policy.infer(obs, noise=noise)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        args.output.with_suffix(".profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35))


if __name__ == "__main__":
    main()
