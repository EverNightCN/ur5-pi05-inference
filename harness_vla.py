"""Run HarnessVLA's RPent API planner against the UR5 pi0.5 controller."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import importlib.metadata
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback
import time

from dotenv import load_dotenv


def parse_args():
    from policy_rollout import RolloutConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['record', 'dry-run'], default='dry-run')
    parser.add_argument('--ur5-repo', default=os.environ.get('UR5_REPO', str(Path(__file__).resolve().parent)))
    parser.add_argument('--config', default=os.environ.get('UR5_CONFIG'))
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
                                                    str(Path(os.environ.get('HARNESSVLA_STATE_DIR', Path(__file__).resolve().parent / '.runtime')) / 'memory/ur5'))))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--protocol', choices=['zero-shot', 'explore', 'evaluate'], default='zero-shot')
    parser.add_argument('--snapshot', type=Path, help='Frozen corpus used by evaluation or a matched baseline')
    parser.add_argument('--scene-id', default='reference-0', help='Physical layout identifier, not a simulator RNG seed')
    parser.add_argument('--task-id', default='ur5_pickplace_2obj')
    parser.add_argument('--cell-tag', help='Exploration session memory inbox identifier')
    parser.add_argument('--scene-description', default='')
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
Physical layout is {args.scene_id}. Layout description: {args.scene_description}
Use observe to inspect current global and wrist RGB images and measured robot state.
This run uses protocol {args.protocol}. Read applicable experience through list_dir and
read_text_file in {args.memory_dir}. Bind reference instructions to the current images.
Read/write run notes under {args.output_dir}/notes. Record each useful decision,
its visual preconditions, outcome checks and recovery options as exploration proceeds.
File tools expose this run and its selected memory corpus. Use absolute paths.
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
Save useful run notes in {args.output_dir}/notes. Call finish with the observed outcome and
supporting summary when the task is complete or an unrecoverable failure occurs.
'''


def main():
    from harness_protocol import (RunMemory, capture_evidence, digest, policy_identity,
                                  prepare_memory, read_json, verify_snapshot, write_json)
    env_file = Path(os.environ.get('HARNESSVLA_ENV_FILE',
                                   str(Path.home() / '.config/harnessvla/ur5.env')))
    load_dotenv(env_file, override=False)
    args = parse_args()
    from policy_rollout import UR5Robot, load_rollout_config, prepare_execution
    from rpent.dashboard.events import NullDashboardEventSink
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
    for value in (args.scene_id, args.task_id, *([args.cell_tag] if args.cell_tag else [])):
        if not value or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in value):
            raise ValueError('scene-id and task-id use letters, numbers, underscores and hyphens')
    root = Path(__file__).resolve().parent
    state_root = Path(os.environ.get('HARNESSVLA_STATE_DIR', root / '.runtime'))
    args.output_dir = (args.output_dir or state_root / 'runs' / str(time.time_ns())).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'notes').mkdir()
    init_output_dir(args.output_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    local_policy = args.host in ('127.0.0.1', 'localhost') and args.port == 8000
    checkpoint = Path(os.environ.get('PI05_CHECKPOINT', root / '.runtime/checkpoint-current'))
    checkpoint_sha = None
    if local_policy and (checkpoint / 'conversion_audit.json').exists():
        checkpoint_sha = read_json(checkpoint / 'conversion_audit.json')['files_sha256']['model.safetensors']
    collect = hardware['collect']
    metadata = dict(mode=args.mode, protocol=args.protocol, task_id=args.task_id, scene_id=args.scene_id,
                    scene_description=args.scene_description,
                    cell_tag=args.cell_tag or f'{args.task_id}_{args.scene_id}_{args.output_dir.name}',
                    task=args.task, policy_prompt=args.prompt, model=args.model, base_url=args.base_url,
                    reasoning_effort=args.reasoning_effort, controller=asdict(cfg),
                    max_turns=args.max_turns, max_tokens=args.max_tokens,
                    planner_timeout_s=args.planner_timeout, exploration_memory=str(args.memory_dir),
                    memory_dir=str(args.memory_dir), snapshot_id=None,
                    snapshot_path=str(args.snapshot.resolve()) if args.snapshot else None,
                    policy_endpoint=f'ws://{args.host}:{args.port}',
                    checkpoint=str(checkpoint.resolve()) if local_policy else None,
                    checkpoint_sha256=checkpoint_sha,
                    hardware_contract=dict(home=hardware['home']['joints_deg'], workspace=hardware['safety'],
                        camera_width=collect['camera_width'], camera_height=collect['camera_height'],
                        camera_fps=collect['camera_fps']),
                    implementation_sha256={name:digest(root / name) for name in
                        ['harness_vla.py', 'harness_tools.py', 'harness_protocol.py', 'harness_workflow.py',
                         'policy_rollout.py', 'rollout_io.py']},
                    runtime_versions={name:importlib.metadata.version(name) for name in
                                      ['pydantic-ai-slim', 'pydantic-ai-harness', 'openai', 'websockets']})
    write_json(args.output_dir / 'configuration.json', metadata)
    memory = None
    snapshot = None
    events = NullDashboardEventSink()
    report = dict(finish=None, stats={}, error=None, mode=args.mode, protocol=args.protocol,
                  physical_success=None, autonomous_success=None, initial_evidence=None, final_evidence=None)
    started = time.monotonic()
    stop = threading.Event()
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        stop.set()
        raise KeyboardInterrupt

    def save_final(robot):
        try:
            report['final_evidence'] = capture_evidence(robot, args.output_dir / 'final')
        except Exception as error:
            logging.exception('Final observation capture failed')
            report['final_capture_error'] = str(error)
            report['error'] = report['error'] or f'Final observation capture failed: {error}'

    signal.signal(signal.SIGTERM, terminate)
    interrupted = False
    try:
        if local_policy:
            subprocess.run(['bash', str(root / 'start_inference.sh'), 'start'], check=True)
        if args.protocol != 'zero-shot' or args.snapshot:
            identity = policy_identity(args.host, args.port, cfg.timeout)
            if metadata['checkpoint_sha256'] and metadata['checkpoint_sha256'] != identity['checkpoint_sha256']:
                raise ValueError('Loaded policy differs from checkpoint-current; restart the model service')
            metadata['checkpoint_sha256'] = identity['checkpoint_sha256']
            metadata['policy_identity'] = identity
        args.memory_dir, snapshot = prepare_memory(args, metadata)
        metadata.update(memory_dir=str(args.memory_dir), snapshot_id=snapshot['id'] if snapshot else None,
                        snapshot_path=str(args.snapshot.resolve()) if args.snapshot else None)
        write_json(args.output_dir / 'configuration.json', metadata)
        memory = RunMemory(args.memory_dir, args.output_dir,
                           cell_tag=metadata['cell_tag'] if args.protocol == 'explore' else None)
        if args.protocol == 'explore':
            (args.memory_dir / '_internal/inbox' / metadata['cell_tag']).mkdir(parents=True, exist_ok=True)
        planner = build_planner('api', output_dir=args.output_dir, recipe_tag='ur5_pickplace',
                                robot_name='ur5', model=args.model, base_url=args.base_url,
                                memory_dir=args.memory_dir, max_tokens=args.max_tokens,
                                reasoning_effort=args.reasoning_effort, dashboard_events=events,
                                planner_timeout_s=args.planner_timeout or math.inf)
        with ExitStack() as stack:
            robot = UR5Robot(args, cfg, hardware, stack)
            stack.callback(save_final, robot)
            stack.callback(robot.hold)
            prepare_execution(robot, args, hardware, cfg, stop)
            report['initial_evidence'] = capture_evidence(robot, args.output_dir / 'initial')
            toolkit = UR5HarnessToolkit(robot=robot, cfg=cfg, output_dir=args.output_dir,
                                        memory=memory, dashboard_events=events)
            stack.callback(toolkit.close)
            solve_started = time.monotonic()
            prompt = system_prompt(args, cfg)
            if args.protocol == 'explore':
                inbox = args.memory_dir / '_internal/inbox' / metadata['cell_tag']
                prompt += f'\nRead exploration history in {inbox} before acting. Write experience drafts there.\n'
            result = planner.solve(system_prompt=prompt,
                                   user_message=f'Execute the task. Start with observe. Task: {args.task}',
                                   toolkit=toolkit, max_turns=args.max_turns or sys.maxsize)
            report.update(finish=result.finish_result, stats=result.stats, error=result.error,
                          planner_elapsed_s=time.monotonic() - solve_started)
            write_json(args.output_dir / 'transcript.json', result.messages)
    except KeyboardInterrupt:
        interrupted = True
        report.update(interrupted=True, error='Run interrupted by operator')
        write_json(args.output_dir / 'interrupted.json', dict(interrupted=True, mode=args.mode))
    except Exception as error:
        logging.exception('Harness run failed')
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        report['elapsed_s'] = time.monotonic() - started
        report['memory_reads'] = memory.reads if memory else []
        if snapshot:
            try:
                verify_snapshot(args.snapshot)
            except (OSError, ValueError) as error:
                report['error'] = f'Frozen memory validation failed: {error}'
        write_json(args.output_dir / 'result.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'Run directory: {args.output_dir}')
    if interrupted:
        return 130
    return 0 if not report['error'] and (report['finish'] or {}).get('status') == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
