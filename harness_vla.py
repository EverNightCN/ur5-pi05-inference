"""Run HarnessVLA's RPent API planner against the UR5 pi0.5 controller."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv


def parse_args():
    from policy_rollout import RolloutConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['record', 'dry-run'], default='dry-run')
    parser.add_argument('--ur5-repo', default=os.environ.get('UR5_REPO', str(Path(__file__).resolve().parent)))
    parser.add_argument('--config')
    parser.add_argument('--robot-ip')
    parser.add_argument('--host', default='127.0.0.1', help='pi0.5 service host')
    parser.add_argument('--port', type=int, default=8000, help='pi0.5 service port')
    parser.add_argument('--global-dev')
    parser.add_argument('--wrist-dev')
    parser.add_argument('--gripper-dev')
    parser.add_argument('--task', help='Overall instruction for the planner')
    parser.add_argument('--policy-prompt', help='Default instruction for the frozen pi0.5 policy')
    parser.add_argument('--model', default=os.environ.get('HARNESSVLA_MODEL'),
                        help='RPent API model id, for example openai-chat:<vision-model>')
    parser.add_argument('--base-url', default=os.environ.get('HARNESSVLA_BASE_URL'))
    parser.add_argument('--memory-dir', type=Path,
                        default=Path(os.environ.get('HARNESSVLA_MEMORY_DIR',
                                                    str(Path(__file__).resolve().parent / '.runtime/memory/ur5'))))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-turns', type=int, default=0, help='Model request budget, 0 means unlimited')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Maximum output tokens per request')
    parser.add_argument('--planner-timeout', type=float, default=0,
                        help='Whole planner run timeout in seconds, 0 means unlimited')
    parser.add_argument('--reasoning-effort', choices=['none', 'low', 'medium', 'high', 'xhigh'],
                        default=os.environ.get('HARNESSVLA_REASONING_EFFORT', 'none'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--i-understand', action='store_true')
    parser.add_argument('--home-at-start', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check dependencies and config without opening hardware')
    for name in RolloutConfig.__dataclass_fields__:
        if name == 'duration':
            continue
        parser.add_argument('--' + name.replace('_', '-'), type=float,
                            help='Override the UR5 [inference] controller setting')
    parser.set_defaults(duration=None)
    args = parser.parse_args()
    if args.mode == 'record' and not (args.execute and args.i_understand):
        parser.error('Motion requires --execute --i-understand')
    if args.home_at_start and args.mode != 'record':
        parser.error('--home-at-start requires --mode record')
    if not args.check and not args.model:
        parser.error('Set --model or HARNESSVLA_MODEL to an image-capable API model')
    if args.max_turns < 0 or args.max_tokens <= 0:
        parser.error('max-turns must be >= 0 and max-tokens must be > 0')
    if not 0 < args.port < 65536:
        parser.error('port must be between 1 and 65535')
    if not math.isfinite(args.planner_timeout) or args.planner_timeout < 0:
        parser.error('planner-timeout must be finite and >= 0')
    args.no_gripper_read = False
    return args


def system_prompt(args, cfg):
    return f'''You operate a UR5 using HarnessVLA's memory-guided planning and action primitives.
The task is {args.task}
Use observe to inspect current global and wrist RGB images and measured robot state.
Review available experience through list_dir and read_text_file in {args.memory_dir}.
Use the current images to bind targets and check the physical outcome after each action.
Use pi05_act for short contact-rich attempts. Use move_delta and rotate_delta for
measured base-frame staging, and set_gripper for explicit opening or closing.
The pi0.5 checkpoint was trained on this complete task instruction: {args.prompt}
Its single-object instruction following has not been evaluated. The default pi05_act
instruction continues the trained task. Pass a custom instruction when that is intended.
Pose adjustments preserve the gripper command. Translation uses millimetres, rotations
use a base-frame axis-angle vector in degrees. State TCP coordinates use metres and radians.
The controller runs at {cfg.fps:g} Hz with {cfg.max_step_mm:g} mm and
{cfg.max_rot_step_deg:g} degree per-tick steps and a shared pose/gripper waypoint timeline.
Dry-run tools report simulated execution against measured observations.
Save useful run notes in {args.output_dir}. Call finish with the observed outcome and
supporting summary when the task is complete or an unrecoverable failure occurs.
'''


def main():
    env_file = Path(os.environ.get('HARNESSVLA_ENV_FILE',
                                   str(Path.home() / '.config/harnessvla/ur5.env')))
    load_dotenv(env_file, override=False)
    args = parse_args()
    from policy_rollout import UR5Robot, load_rollout_config, prepare_execution
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.planner.base import build_planner
    from rpent.utils.logging import init_output_dir
    from harness_tools import UR5HarnessToolkit

    hardware, cfg = load_rollout_config(args)
    args.prompt = args.policy_prompt or hardware['collect']['task_en']
    args.task = args.task or args.prompt
    args.memory_dir = args.memory_dir.expanduser().resolve()
    if args.check:
        print(json.dumps(dict(dependencies='imported', controller=asdict(cfg),
                              home_degrees=hardware['home']['joints_deg'],
                              model=args.model, reasoning_effort=args.reasoning_effort,
                              api_request_tested=False, hardware_opened=False), indent=2))
        return 0
    args.output_dir = (args.output_dir or (Path(__file__).resolve().parent / '.runtime/runs') /
                       str(time.time_ns())).expanduser().resolve()
    init_output_dir(args.output_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args.memory_dir.mkdir(parents=True, exist_ok=True)
    events = NullDashboardEventSink()
    planner = build_planner('api', output_dir=args.output_dir, recipe_tag='ur5_pickplace',
                            robot_name='ur5', model=args.model, base_url=args.base_url,
                            memory_dir=args.memory_dir, max_tokens=args.max_tokens,
                            reasoning_effort=args.reasoning_effort, dashboard_events=events,
                            planner_timeout_s=args.planner_timeout or math.inf)
    local_policy = args.host in ('127.0.0.1', 'localhost') and args.port == 8000
    metadata = dict(mode=args.mode, task=args.task, policy_prompt=args.prompt, model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    memory_dir=str(args.memory_dir), controller=asdict(cfg),
                    max_turns=args.max_turns, planner_timeout_s=args.planner_timeout,
                    policy_endpoint=f'ws://{args.host}:{args.port}',
                    checkpoint=str((Path(__file__).parent / '.runtime/checkpoint-current').resolve()) if local_policy else None)
    (args.output_dir / 'configuration.json').write_text(json.dumps(metadata, indent=2))
    if local_policy:
        subprocess.run(['bash', str(Path(__file__).parent / 'start_inference.sh'), 'start'], check=True)
    stop = threading.Event()
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        with ExitStack() as stack:
            robot = UR5Robot(args, cfg, hardware, stack)
            stack.callback(robot.hold)
            prepare_execution(robot, args, hardware, cfg, stop)
            toolkit = UR5HarnessToolkit(robot=robot, cfg=cfg, output_dir=args.output_dir,
                                        memory=MemoryManager(args.memory_dir), dashboard_events=events)
            stack.callback(toolkit.close)
            result = planner.solve(system_prompt=system_prompt(args, cfg),
                                   user_message=f'Execute the task. Start with observe. Task: {args.task}',
                                   toolkit=toolkit, max_turns=args.max_turns or sys.maxsize)
            report = dict(finish=result.finish_result, stats=result.stats, error=result.error,
                          mode=args.mode, physical_success=None)
            (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2))
            (args.output_dir / 'transcript.json').write_text(json.dumps(result.messages, indent=2, default=str))
            print(json.dumps(report, indent=2))
            if result.error:
                raise RuntimeError(result.error)
            return 0 if (result.finish_result or {}).get('status') == 'success' else 1
    except KeyboardInterrupt:
        (args.output_dir / 'interrupted.json').write_text(json.dumps(dict(interrupted=True, mode=args.mode)))
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    raise SystemExit(main())
