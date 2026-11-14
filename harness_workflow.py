"""Operator-driven exploration, publication and matched frozen-memory evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from dotenv import load_dotenv
from harness_protocol import (MOTION_TOOLS, SUCCESS_CONDITIONS, RunMemory, build_artifacts, contract,
                              freeze_memory, publish_run, read_calls, read_json,
                              review_run, verify_snapshot, write_json)

ROOT = Path(__file__).resolve().parent
STATE_ROOT = Path(os.environ.get('HARNESSVLA_STATE_DIR', ROOT / '.runtime'))


def confirm(message):
    return input(message + ' [y/N] ').strip().lower() == 'y'


def operator_review(directory, outcome=None, notes='', confirmed=False, interventions=0):
    print(f'最终图像与状态证据目录: {Path(directory).resolve() / "final"}')
    if outcome is None:
        outcome = input('实际结果 success / failure / hardware-error / interrupted: ').strip()
        notes = input('验收说明或失败原因: ').strip()
        interventions = int(input('本轮执行中的人工干预次数（初态复位不计）[0]: ').strip() or '0')
    conditions = {key: False for key in SUCCESS_CONDITIONS}
    if outcome == 'success':
        if confirmed:
            conditions = dict.fromkeys(SUCCESS_CONDITIONS, True)
        else:
            conditions = dict(zip(SUCCESS_CONDITIONS, [
                confirm('红色圆柱和蓝色方块均已在盒内？'),
                confirm('其他物体均在盒外？'),
                confirm('夹爪已松开并撤离，目标物体仍留在盒内？')]))
    return review_run(directory, outcome=outcome, notes=notes, conditions=conditions,
                      interventions=interventions)


def distill(directory):
    """Use recorded evidence and file tools to annotate executed actions after real success."""
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.planner.base import build_planner
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.logging import init_output_dir
    directory = Path(directory).resolve()
    configuration = read_json(directory / 'configuration.json')
    review = read_json(directory / 'review.json')
    if review['physical_success'] is not True or configuration['protocol'] != 'explore':
        raise ValueError('Distillation requires an operator-confirmed successful exploration')
    env_file = Path(os.environ.get('HARNESSVLA_ENV_FILE', Path.home() / '.config/harnessvla/ur5.env'))
    load_dotenv(env_file, override=False)
    init_output_dir(directory)
    calls = [{key:item[key] for key in ('call_id', 'tool', 'arguments')}
             for item in read_calls(directory) if item['tool'] in MOTION_TOOLS and not item['error']]
    if not calls:
        raise ValueError('No successful motion calls to distill')
    memory = RunMemory(directory / 'distillation-memory', directory)
    events = NullDashboardEventSink()
    toolkit = Toolkit(dashboard_events=events, memory=memory)
    planner = build_planner('api', output_dir=directory, recipe_tag='ur5_distillation', robot_name='ur5',
        model=configuration['model'], base_url=configuration['base_url'],
        max_tokens=configuration['max_tokens'], reasoning_effort=configuration['reasoning_effort'],
        planner_timeout_s=configuration['planner_timeout_s'] or math.inf, dashboard_events=events)
    prompt = f'''Distill this operator-confirmed successful UR5 run into a reusable strategy.
Use file tools only. Read {directory}/configuration.json, result.json, review.json,
tool-calls.jsonl, states.json and notes as needed. The complete initial/final evidence
and tool outcomes are already saved in this run. Ground every claim in those records.
Write {directory}/notes/strategy.json with this schema:
{{"applicability":"initial visual/robot conditions where this strategy applies",
  "lessons":["evidence-backed execution or recovery lessons"],
  "steps":[{{"call_id":1,"when":"visual preconditions for this step",
    "verify":"observable completion criteria", "on_failure":"recovery decision",
    "rebind":"how to derive new target/relative displacement from current observations"}}]}}
Include exactly one step per successful motion call listed below, preserving call_id order.
Arguments will be taken from the execution journal by the publisher. Reference arguments
are examples from this run; specify how to adapt to a changed object layout in rebind.
Distinguish observed facts from uncertain interpretations. Save strategy.json before finish.
Successful motion calls: {json.dumps(calls, ensure_ascii=False)}'''
    started = time.monotonic()
    try:
        result = planner.solve(system_prompt=prompt, user_message='Read the run, write the strategy, then finish.',
                               toolkit=toolkit, max_turns=configuration['max_turns'] or sys.maxsize)
        write_json(directory / 'distillation.json', dict(stats=result.stats, error=result.error,
                   elapsed_s=time.monotonic() - started, finish=result.finish_result))
        write_json(directory / 'distillation-transcript.json', result.messages)
        if result.error:
            raise RuntimeError(result.error)
        if (result.finish_result or {}).get('status') != 'success':
            raise RuntimeError('Distillation did not finish successfully')
        build_artifacts(directory)
    finally:
        toolkit.close()


def child_run(arguments):
    process = subprocess.Popen([sys.executable, str(ROOT / 'harness_vla.py'), *arguments])
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait()
        return 130


def explore(args, extra):
    session = (args.output_dir or STATE_ROOT / 'exploration' / str(time.time_ns())).resolve()
    session.mkdir(parents=True, exist_ok=False)
    description = args.reference_description or input('描述参考初态的物体、盒子及相机布置: ').strip()
    if not description:
        raise ValueError('Reference layout description is required')
    cell_tag = f'ur5_{args.scene_id}_{time.time_ns()}'
    write_json(session / 'session.json', dict(scene_id=args.scene_id, reference_description=description, cell_tag=cell_tag,
               reset_method='operator restores the physical scene before each attempt', attempts=[]))
    index = 0
    while args.attempts == 0 or index < args.attempts:
        answer = input(f'恢复参考初态 {args.scene_id}（含桌面物体），输入 ready 开始，q 退出: ').strip()
        if answer == 'q':
            return 0
        if answer != 'ready':
            print('等待明确的初态复位确认。')
            continue
        index += 1
        directory = session / f'attempt-{index:03d}'
        manifest = read_json(session / 'session.json')
        manifest['attempts'].append(dict(directory=str(directory), reset_confirmed_at=time.time()))
        write_json(session / 'session.json', manifest)
        status = child_run(['--mode', 'record', '--home-at-start', '--execute', '--i-understand',
                            *extra, '--protocol', 'explore', '--scene-id', args.scene_id,
                            '--cell-tag', cell_tag, '--scene-description', description,
                            '--memory-dir', str(args.memory_dir.resolve()), '--output-dir', str(directory)])
        print(f'Attempt saved at {directory}, exit={status}')
        if not (directory / 'result.json').exists():
            raise RuntimeError('Attempt failed before a result could be recorded; inspect the reported error')
        if status == 130:
            print(f'本轮已中断，可使用 review {directory} 独立验收。')
            return 130
        review = operator_review(directory)
        inbox = args.memory_dir.resolve() / '_internal/inbox' / cell_tag
        history_path = inbox / 'attempts.json'
        history = read_json(history_path) if history_path.exists() else []
        result = read_json(directory / 'result.json')
        notes = {str(path.relative_to(directory)):path.read_text() for path in (directory / 'notes').rglob('*') if path.is_file()}
        history.append(dict(attempt=index, result=result, operator_review=review, notes=notes))
        write_json(history_path, history)
        if review['autonomous_success']:
            distill(directory)
            receipt = publish_run(directory)
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            return 0
    return 1


def evaluate(args, extra):
    verify_snapshot(args.snapshot)
    answer = input(f'按评测布置 {args.scene_id} 恢复桌面，输入 ready 开始，q 退出: ').strip()
    if answer == 'q':
        return 0
    if answer != 'ready':
        raise ValueError('Evaluation requires explicit scene reset confirmation')
    directory = (args.output_dir or STATE_ROOT / 'evaluation' / str(time.time_ns())).resolve()
    status = child_run(['--mode', 'record', '--home-at-start', '--execute', '--i-understand', *extra,
                       '--protocol', 'zero-shot' if args.zero_shot else 'evaluate',
                       '--snapshot', str(args.snapshot.resolve()), '--scene-id', args.scene_id,
                       '--output-dir', str(directory)])
    print(f'Run saved at {directory}, exit={status}')
    if status == 130:
        print(f'本轮已中断，可使用 review {directory} 独立验收。')
        return status
    if not (directory / 'result.json').exists():
        raise RuntimeError('Evaluation configuration or initialization failed')
    verdict = operator_review(directory)
    return 0 if verdict['autonomous_success'] else 1


def summarize(directory):
    rows = []
    for path in sorted(Path(directory).rglob('result.json')):
        if not (path.parent / 'configuration.json').exists():
            continue
        configuration, result = read_json(path.parent / 'configuration.json'), read_json(path)
        calls = read_calls(path.parent) if (path.parent / 'tool-calls.jsonl').exists() else []
        review = read_json(path.parent / 'review.json') if (path.parent / 'review.json').exists() else None
        distillation = read_json(path.parent / 'distillation.json') if (path.parent / 'distillation.json').exists() else {}
        rows.append(dict(run=str(path.parent), protocol=configuration['protocol'],
            contract_id=hashlib.sha256(json.dumps(contract(configuration),sort_keys=True).encode()).hexdigest(),
            scene_id=configuration['scene_id'], snapshot_id=configuration['snapshot_id'],
            mode=configuration['mode'], model=configuration['model'],
            physical_success=review['physical_success'] if review else None,
            autonomous_success=review['autonomous_success'] if review else None,
            outcome=review['outcome'] if review else 'pending',
            interventions=review['interventions'] if review else None,
            elapsed_s=result['elapsed_s'], planner_elapsed_s=result.get('planner_elapsed_s'),
            vla_calls=sum(call['tool'] == 'pi05_act' for call in calls),
            api_usage=result['stats'], distillation_usage=distillation.get('stats'),
            error=result['error']))
    groups = []
    keys = sorted({(row['mode'], row['protocol'], row['model'], row['snapshot_id'] or '',row['contract_id']) for row in rows})
    for mode,protocol,model,snapshot,configuration_id in keys:
        runs = [row for row in rows if (row['mode'],row['protocol'],row['model'],row['snapshot_id'] or '',row['contract_id']) == (mode,protocol,model,snapshot,configuration_id)]
        reviewed = [row for row in runs if row['physical_success'] is not None]
        groups.append(dict(mode=mode, protocol=protocol, model=model, snapshot_id=snapshot or None, contract_id=configuration_id,
            attempts=len(runs), reviewed=len(reviewed), pending=len(runs)-len(reviewed),
            successes=sum(row['autonomous_success'] is True for row in reviewed),
            success_rate=sum(row['autonomous_success'] is True for row in reviewed)/len(runs) if len(reviewed)==len(runs) else None,
            physical_success_rate=sum(row['physical_success'] is True for row in reviewed)/len(runs) if len(reviewed)==len(runs) else None,
            median_elapsed_s=statistics.median(row['elapsed_s'] for row in runs),
            hardware_errors=sum(row['outcome'] == 'hardware-error' for row in reviewed)))
    return dict(runs=rows, groups=groups,
                comparison='Compare reviewed trials with the same snapshot_id and matched scene IDs; exploration costs are separate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    exploration = commands.add_parser('explore')
    exploration.add_argument('--memory-dir', type=Path, default=STATE_ROOT / 'memory/ur5')
    exploration.add_argument('--output-dir', type=Path)
    exploration.add_argument('--scene-id', default='reference-0')
    exploration.add_argument('--reference-description')
    exploration.add_argument('--attempts', type=int, default=0, help='0 means continue until success or operator quit')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--snapshot', type=Path, required=True)
    evaluation.add_argument('--scene-id', required=True)
    evaluation.add_argument('--output-dir', type=Path)
    evaluation.add_argument('--zero-shot', action='store_true')
    review = commands.add_parser('review')
    review.add_argument('run', type=Path)
    review.add_argument('--outcome', choices=['success','failure','hardware-error','interrupted'])
    review.add_argument('--notes', default='')
    review.add_argument('--conditions-confirmed', action='store_true', help='Attest all three physical success conditions')
    review.add_argument('--interventions', type=int, default=0)
    for name in ['distill', 'publish']:
        commands.add_parser(name).add_argument('run', type=Path)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--memory-dir', type=Path, default=STATE_ROOT / 'memory/ur5')
    freeze.add_argument('--output-dir', type=Path, required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('directory', type=Path, nargs='?', default=STATE_ROOT)
    summary.add_argument('--output', type=Path)
    args, extra = parser.parse_known_args()
    if extra and args.command not in {'explore', 'evaluate'}:
        parser.error(f'Unrecognized arguments: {extra}')
    if args.command == 'explore':
        if args.attempts < 0:
            parser.error('--attempts must be nonnegative')
        return explore(args, extra)
    if args.command == 'evaluate':
        return evaluate(args, extra)
    if args.command == 'review':
        result = operator_review(args.run, args.outcome, args.notes, args.conditions_confirmed, args.interventions)
    elif args.command == 'distill':
        distill(args.run)
        return 0
    elif args.command == 'publish':
        result = publish_run(args.run)
    elif args.command == 'freeze':
        result = freeze_memory(args.memory_dir, args.output_dir)
    else:
        result = summarize(args.directory)
        if args.output:
            write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
