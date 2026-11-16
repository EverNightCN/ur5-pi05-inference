"""One observe -> pi0.5 segment -> observe round for a caller-supplied planner."""
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner_bridge import open_session
from policy_rollout import parse_args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.mode == 'home':
        raise ValueError('Use run_robot.sh home for standalone HOME motion')
    with open_session(args) as session:
        before = session.observe()
        print(json.dumps(dict(stage='before', feedback=before['feedback'],
                              image_shape=list(before['observation']['observation/image'].shape))), flush=True)
        if args.mode != 'observe':
            seconds = session.cfg.duration if session.cfg.duration > 0 else 4.0
            result = session.act(args.prompt, seconds=seconds)
            after = session.observe()
            print(json.dumps(dict(execution=result, feedback=after['feedback'])), flush=True)
            # Pass after['observation'] to VeraPlan's condition/outcome checker here.


if __name__ == '__main__':
    main()
