"""Persistent local inference endpoint for the complete UR5 pi0.5 checkpoint."""
import argparse
import logging
import json
import time
from pathlib import Path

from runtime import load_policy, read_observation
import torch
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmup-sample", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--compile-all", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-prompt-from-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.warmup_runs < 1:
        parser.error("--warmup-runs must be positive")
    logging.basicConfig(level=logging.INFO, force=True)
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint, attention_backend=args.attention,
                         compile_mode=args.compile_mode, cuda_graph=True, compile_all=args.compile_all,
                         cache_padding=True)
    observation = read_observation(args.warmup_sample)
    if args.warmup_prompt_from_checkpoint:
        spec = json.loads((args.checkpoint / "source_model_spec.json").read_text())
        prompt = spec["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Checkpoint task prompt is empty")
        observation["prompt"] = prompt
    for index in range(args.warmup_runs):
        started = time.monotonic()
        policy.infer(observation)
        torch.cuda.synchronize()
        logging.info("Model warmup %d/%d completed in %.3fs", index + 1,
                     args.warmup_runs, time.monotonic() - started)
    logging.info("Complete pi0.5 inference is ready on 127.0.0.1:%s", args.port)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
