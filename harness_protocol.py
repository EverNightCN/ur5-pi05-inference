"""Evidence, operator verdicts and versioned memory for real-robot Harness runs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from rpent.memory import MemoryManager

MOTION_TOOLS = {'pi05_act', 'move_delta', 'rotate_delta', 'set_gripper'}
SUCCESS_CONDITIONS = ('targets_in_box', 'other_objects_outside', 'released_and_retreated')
CONTRACT_KEYS = ('task_id', 'task', 'policy_prompt', 'model', 'base_url', 'reasoning_effort',
                 'controller', 'max_turns', 'max_tokens', 'planner_timeout_s', 'checkpoint_sha256',
                 'hardware_contract', 'implementation_sha256', 'runtime_versions')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Keep each state document complete across interruption and concurrent readers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.protocol.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


class RunMemory(MemoryManager):
    """Expose this run's files and selected corpus through the native file tools."""
    def __init__(self, root, output_dir, *, cell_tag=None):
        super().__init__(root, memory_access='inbox_write' if cell_tag else 'read_only',
                         inbox_cell_tag=cell_tag)
        self.output_dir = Path(output_dir).resolve()
        self.reads = []

    def _authorize(self, path, *, write):
        resolved = super()._authorize(path, write=write)
        if resolved.is_relative_to(self.root):
            return resolved
        if not resolved.is_relative_to(self.output_dir):
            raise PermissionError(f'File is outside this run and its selected memory: {path}')
        if write and not resolved.is_relative_to(self.output_dir / 'notes'):
            raise PermissionError('Agent notes and strategy drafts belong under <output_dir>/notes/')
        return resolved

    def record_read(self, path, content):
        self.reads.append(dict(path=str(path), content_sha256=hashlib.sha256(content.encode()).hexdigest()))


def capture_evidence(robot, directory):
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    observation, observed_at = robot.get_observation()
    for key, filename in [('observation/image', 'global.png'), ('observation/wrist_image', 'wrist.png')]:
        Image.fromarray(observation[key]).save(directory / filename)
    write_json(directory / 'state.json', dict(observed_at=observed_at,
               state=observation['observation/state'].tolist(), captured_at=time.time()))
    return {name: digest(directory / name) for name in ['global.png', 'wrist.png', 'state.json']}


def policy_identity(host, port, timeout):
    """Read the policy server's actual loaded-checkpoint identity without issuing actions."""
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    uri = host if host.startswith('ws') else f'ws://{host}'
    with connect(f'{uri}:{port}', open_timeout=timeout, close_timeout=1,
                 compression=None, max_size=None) as connection:
        identity = msgpack_numpy.unpackb(connection.recv(timeout=timeout))
    if identity.get('action_horizon') != 50 or identity.get('num_steps') != 10:
        raise ValueError(f'Unexpected full-policy metadata: {identity}')
    checksum = identity.get('checkpoint_sha256', '')
    if len(checksum) != 64 or any(character not in '0123456789abcdef' for character in checksum):
        raise ValueError('The policy endpoint must identify its loaded checkpoint SHA-256')
    return identity


def contract(configuration):
    return {key: configuration[key] for key in CONTRACT_KEYS}


def file_manifest(directory, *, exclude=()):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Snapshot contains a symbolic link: {path}')
        if path.is_file() and path.relative_to(directory).as_posix() not in exclude:
            files[path.relative_to(directory).as_posix()] = digest(path)
    return files


def verify_snapshot(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'snapshot.json')
    if file_manifest(directory, exclude=('snapshot.json',)) != manifest['files']:
        raise ValueError('Frozen memory file set or contents changed')
    identity = hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k != 'id'},
                                       sort_keys=True).encode()).hexdigest()
    if identity != manifest['id']:
        raise ValueError('Frozen memory manifest changed')
    return manifest


def prepare_memory(args, configuration):
    """Bind an explicit frozen corpus, exploration workspace or empty baseline corpus."""
    if args.protocol == 'explore':
        return args.memory_dir, None
    if args.protocol == 'evaluate' and args.snapshot is None:
        raise ValueError('Memory evaluation requires --snapshot')
    snapshot = verify_snapshot(args.snapshot) if args.snapshot else None
    if snapshot and contract(configuration) != snapshot['contract']:
        changed = [key for key in CONTRACT_KEYS if configuration[key] != snapshot['contract'][key]]
        raise ValueError(f'Evaluation differs from frozen configuration: {changed}')
    if args.protocol == 'evaluate':
        if not any(name.startswith('task-specific/') and name.endswith('_recipe.jsonl')
                   for name in snapshot['files']):
            raise ValueError('Snapshot contains no successful task recipe')
        return args.snapshot.resolve(), snapshot
    empty = args.output_dir / 'empty-memory'
    empty.mkdir()
    return empty, snapshot


def review_run(directory, *, outcome, notes, conditions, interventions=0):
    directory = Path(directory).resolve()
    if outcome not in {'success', 'failure', 'hardware-error', 'interrupted'}:
        raise ValueError('Unknown operator outcome')
    if interventions < 0:
        raise ValueError('Interventions must be nonnegative')
    with locked(directory):
        configuration = read_json(directory / 'configuration.json')
        result = read_json(directory / 'result.json')
        if configuration['mode'] != 'record':
            raise ValueError('Dry-run cannot receive a physical task verdict')
        evidence = result.get('final_evidence')
        if outcome == 'success':
            if not all(conditions.get(key) is True for key in SUCCESS_CONDITIONS):
                raise ValueError('Success requires confirmation of all three physical conditions')
            if not evidence or any(digest(directory / 'final' / name) != value for name,value in evidence.items()):
                raise ValueError('Final visual/state evidence is missing or changed')
        judgement = dict(outcome=outcome, physical_success=outcome == 'success', notes=notes,
                         autonomous_success=outcome == 'success' and interventions == 0,
                         conditions=conditions, interventions=interventions,
                         evidence=evidence, operator=getpass.getuser())
        path = directory / 'review.json'
        if path.exists():
            existing = read_json(path)
            if any(existing.get(key) != value for key,value in judgement.items()):
                raise ValueError('A different operator verdict is already recorded for this attempt')
            judgement = existing
        else:
            judgement['reviewed_at'] = time.time()
            write_json(path, judgement)
        result.update(physical_success=judgement['physical_success'], operator_outcome=outcome,
                      autonomous_success=judgement['autonomous_success'], interventions=interventions)
        write_json(directory / 'result.json', result)
        return judgement


def read_calls(directory):
    path = Path(directory) / 'tool-calls.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_artifacts(directory):
    directory = Path(directory).resolve()
    configuration = read_json(directory / 'configuration.json')
    review = read_json(directory / 'review.json')
    if configuration['protocol'] != 'explore' or review['physical_success'] is not True:
        raise ValueError('Publishing requires an operator-confirmed successful exploration attempt')
    if review['autonomous_success'] is not True:
        raise ValueError('An assisted attempt cannot establish an autonomous task recipe')
    if not configuration.get('checkpoint_sha256'):
        raise ValueError('Successful task memory requires an identified policy checkpoint')
    if file_manifest(directory / 'final') != review['evidence']:
        raise ValueError('Final evidence changed after operator review')
    calls = read_calls(directory)
    successful = {item['call_id']: item for item in calls
                  if item['tool'] in MOTION_TOOLS and not item['error']}
    if not successful:
        raise ValueError('No successful physical tool calls were recorded')
    draft = read_json(directory / 'notes/strategy.json')
    if not str(draft.get('applicability', '')).strip() or not isinstance(draft.get('steps'), list):
        raise ValueError('Strategy requires applicability and steps')
    seen = set()
    recipe = []
    for step in draft['steps']:
        call_id = step['call_id']
        if call_id not in successful or call_id in seen:
            raise ValueError(f'Strategy references an invalid or duplicate successful call: {call_id}')
        for key in ('when', 'verify', 'on_failure', 'rebind'):
            if not isinstance(step.get(key), str) or not step[key].strip():
                raise ValueError(f'Strategy call {call_id} requires {key}')
        call = successful[call_id]
        recipe.append(dict(call_id=call_id, action=call['tool'], reference_arguments=call['arguments'],
                           when=step['when'], verify=step['verify'], on_failure=step['on_failure'],
                           rebind=step['rebind']))
        seen.add(call_id)
    if seen != set(successful):
        raise ValueError(f'Strategy omits successful motion calls: {sorted(set(successful)-seen)}')
    if [item['call_id'] for item in recipe] != sorted(seen):
        raise ValueError('Recipe must preserve the recorded action order')
    cell = configuration['cell_tag']
    audit = dict(task_complete=True, physical_success=True, configuration=configuration,
                 operator_review=review, applicability=draft['applicability'],
                 lessons=draft.get('lessons', []), tools=calls,
                 provenance=file_manifest(directory / 'initial'),
                 final_evidence=review['evidence'], source_run=str(directory))
    result = read_json(directory / 'result.json')
    if file_manifest(directory / 'initial') != result.get('initial_evidence'):
        raise ValueError('Initial evidence is missing or changed')
    write_json(directory / f'{cell}.json', audit)
    # Write the JSONL through an atomic file replacement as well.
    temporary = directory / f'.{cell}_recipe.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in recipe))
    temporary.replace(directory / f'{cell}_recipe.jsonl')
    return configuration, audit


def publish_run(directory):
    directory = Path(directory).resolve()
    with locked(directory):
        configuration, audit = build_artifacts(directory)
        memory = Path(configuration['exploration_memory'])
        cell = configuration['cell_tag']
        with locked(memory):
            pair = [f'{cell}.json', f'{cell}_recipe.jsonl']
            targets = [memory / 'task-specific' / name for name in pair]
            present = [target.exists() for target in targets]
            if any(present):
                if any(target.exists() and digest(target) != digest(directory / name)
                       for target,name in zip(targets,pair)):
                    raise ValueError('Existing published task pair is incomplete or differs from this run')
                # Complete a matching half-published pair after a previous interrupted copy.
                for target,name in zip(targets,pair):
                    if not target.exists():
                        temporary = target.with_name('.' + target.name + '.pending')
                        shutil.copyfile(directory / name, temporary)
                        temporary.replace(target)
            MemoryManager(memory).merge_memory(cell_tag=cell, run_state_dir=directory, solved=True)
            evidence_files = {}
            for phase in ['initial', 'final']:
                for source in sorted((directory / phase).iterdir()):
                    relative = f'evidence/{cell}/{phase}/{source.name}'
                    target = memory / 'task-specific' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and digest(target) != digest(source):
                        raise ValueError(f'Published evidence differs: {relative}')
                    shutil.copyfile(source, target)
                    evidence_files[relative] = digest(target)
            # Hash the complete evidence-backed pair so freezing can validate its provenance.
            receipt = dict(run=str(directory), contract=contract(configuration),
                           files={name:digest(memory / 'task-specific' / name) for name in pair},
                           reviewed_at=audit['operator_review']['reviewed_at'])
            receipt['files'].update(evidence_files)
            write_json(memory / 'results' / f'{cell}.json', receipt)
        write_json(directory / 'publication.json', receipt)
        return receipt


def freeze_memory(memory, destination):
    memory, destination = Path(memory).resolve(), Path(destination).resolve()
    if destination == memory or destination.is_relative_to(memory):
        raise ValueError('Frozen snapshot must be outside the exploration workspace')
    with locked(memory):
        receipts = [read_json(path) for path in sorted((memory / 'results').glob('*.json'))]
        if not receipts:
            raise ValueError('No published, operator-confirmed successful task memory exists')
        expected_contract = receipts[0]['contract']
        for receipt in receipts:
            if receipt['contract'] != expected_contract:
                raise ValueError('Exploration corpus mixes different model/task/controller configurations')
            for name,checksum in receipt['files'].items():
                if digest(memory / 'task-specific' / name) != checksum:
                    raise ValueError('Published audit/recipe no longer matches its receipt')
        if destination.exists():
            raise FileExistsError('Choose a new snapshot directory for each frozen version')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / 'corpus'
            staged.mkdir()
            # Task pairs from receipts are authoritative, including recovery context in their audits.
            for receipt in receipts:
                for name in receipt['files']:
                    target = staged / 'task-specific' / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(memory / 'task-specific' / name, target)
            for layer in ['global', 'task-family']:
                if (memory / layer).exists():
                    shutil.copytree(memory / layer, staged / layer)
            MemoryManager(staged).rebuild_index()
            manifest = dict(contract=expected_contract, created_at=time.time(), files=file_manifest(staged))
            manifest['id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            write_json(staged / 'snapshot.json', manifest)
            staged.replace(destination)
        return manifest
