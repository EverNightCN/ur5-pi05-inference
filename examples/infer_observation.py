"""Send caller-provided RGB/state arrays and language to the existing model service."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from rollout_io import validate_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.sample, allow_pickle=False) as sample:
        observation = {name: sample[name] for name in (
            'observation/image', 'observation/wrist_image', 'observation/state')}
    observation['prompt'] = args.instruction
    client = WebsocketClientPolicy(args.host, args.port)
    actions = validate_actions(client.infer(observation)['actions'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.save(stream, actions)
    print(f'Model metadata: {client.get_server_metadata()}')
    print(f'Saved actions {actions.shape} to {args.output}')


if __name__ == '__main__':
    main()
