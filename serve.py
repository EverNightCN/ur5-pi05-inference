"""Persistent local inference endpoint for the complete UR5 pi0.5 checkpoint."""
import argparse
import logging
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint, attention_backend=args.attention,
                         compile_mode=args.compile_mode, cuda_graph=True, compile_all=args.compile_all,
                         cache_padding=True)
    policy.infer(read_observation(args.warmup_sample))
    torch.cuda.synchronize()
    logging.info("Complete pi0.5 inference is ready on 127.0.0.1:%s", args.port)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
