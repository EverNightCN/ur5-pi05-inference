"""Real-memory protocol tests with recorded fake observations and no hardware/API calls."""
from __future__ import annotations

from dataclasses import asdict
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_RPENT = find_spec('rpent') is not None
if HAS_RPENT:
    from harness_protocol import (RunMemory, SUCCESS_CONDITIONS, build_artifacts,
        capture_evidence, freeze_memory, prepare_memory, publish_run,
        read_json, review_run, verify_snapshot, write_json)
    from policy_rollout import RolloutConfig
    from test_harness_vla import FakeRobot


@unittest.skipUnless(HAS_RPENT, 'RPent is required in the dedicated Harness environment')
class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'attempt-001'
        self.run.mkdir()
        (self.run / 'notes').mkdir()
        self.memory = self.root / 'memory'
        self.configuration = dict(task_id='pickplace', task='pick two objects', policy_prompt='trained task',
            model='openai-responses:test', base_url='https://api.example/v1', reasoning_effort='low',
            controller=asdict(RolloutConfig()), max_turns=0, max_tokens=8192, planner_timeout_s=0,
            checkpoint_sha256='a'*64, hardware_contract={'home':[-180,-90,-90,-90,90,-90]},
            implementation_sha256={'controller':'v1'}, runtime_versions={'test':'1'}, protocol='explore', mode='record',
            scene_id='reference-0', snapshot_id=None, cell_tag='pickplace_reference_0_attempt001',
            exploration_memory=str(self.memory))
        write_json(self.run / 'configuration.json', self.configuration)
        robot = FakeRobot()
        initial = capture_evidence(robot, self.run / 'initial')
        final = capture_evidence(robot, self.run / 'final')
        write_json(self.run / 'result.json', dict(physical_success=None, initial_evidence=initial,
            final_evidence=final, stats={}, error=None, elapsed_s=1))
        calls = [dict(call_id=1,tool='pi05_act',arguments={'seconds':4},error=None),
                 dict(call_id=2,tool='move_delta',arguments={'dx_mm':2.,'dy_mm':0.,'dz_mm':0.},error=None),
                 dict(call_id=3,tool='pi05_act',arguments={'seconds':2},error='gripper communication fault')]
        (self.run / 'tool-calls.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in calls))
        strategy = dict(applicability='Same task, cameras and gripper configuration', lessons=['Inspect the grasp'],
            steps=[dict(call_id=index,when='Object visible',verify='Object has reached intended state',
                        on_failure='Observe and reposition',rebind='Derive displacement from current object and TCP')
                   for index in [1,2]])
        write_json(self.run / 'notes/strategy.json', strategy)

    def success(self):
        return review_run(self.run, outcome='success', notes='All objects verified after retreat',
                          conditions=dict.fromkeys(SUCCESS_CONDITIONS, True))

    def test_only_operator_review_sets_physical_success(self):
        with self.assertRaisesRegex(ValueError, 'three physical'):
            review_run(self.run, outcome='success', notes='', conditions={})
        self.assertIsNone(read_json(self.run / 'result.json')['physical_success'])
        self.success()
        self.assertTrue(read_json(self.run / 'result.json')['physical_success'])

    def test_review_retries_repair_a_stale_result_from_the_canonical_verdict(self):
        verdict = self.success()
        result = read_json(self.run / 'result.json')
        result['physical_success'] = None
        write_json(self.run / 'result.json', result)
        self.assertEqual(self.success(), verdict)
        self.assertTrue(read_json(self.run / 'result.json')['physical_success'])

    def test_dry_run_cannot_be_accepted_as_physical_success(self):
        self.configuration['mode'] = 'dry-run'
        write_json(self.run / 'configuration.json', self.configuration)
        with self.assertRaisesRegex(ValueError, 'Dry-run'):
            self.success()

    def test_failure_retains_evidence_without_publishing_success(self):
        review_run(self.run, outcome='hardware-error', notes='Modbus exception', conditions={})
        self.assertFalse(read_json(self.run / 'result.json')['physical_success'])
        with self.assertRaisesRegex(ValueError, 'operator-confirmed'):
            publish_run(self.run)
        self.assertFalse((self.memory / 'task-specific').exists())

    def test_assisted_completion_is_retained_but_not_published_as_autonomous_success(self):
        review_run(self.run,outcome='success',notes='Operator moved an object',
                   conditions=dict.fromkeys(SUCCESS_CONDITIONS,True),interventions=1)
        result=read_json(self.run/'result.json')
        self.assertTrue(result['physical_success'])
        self.assertFalse(result['autonomous_success'])
        with self.assertRaisesRegex(ValueError,'assisted attempt'):
            publish_run(self.run)

    def test_final_evidence_must_match_the_operator_review(self):
        self.success()
        (self.run / 'final/global.png').write_bytes(b'changed evidence')
        with self.assertRaisesRegex(ValueError, 'Final evidence'):
            publish_run(self.run)

    def test_initial_evidence_must_match_the_run(self):
        self.success()
        (self.run / 'initial/state.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Initial evidence'):
            publish_run(self.run)

    def test_recipe_uses_factual_arguments_and_covers_successful_calls(self):
        self.success()
        publish_run(self.run)
        path = self.memory / 'task-specific' / (self.configuration['cell_tag'] + '_recipe.jsonl')
        recipe = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([step['call_id'] for step in recipe], [1,2])
        self.assertEqual(recipe[1]['reference_arguments']['dx_mm'], 2.)
        self.assertIn('rebind', recipe[1])

    def test_recipe_rejects_unexecuted_or_failed_calls(self):
        self.success()
        draft = read_json(self.run / 'notes/strategy.json')
        draft['steps'][1]['call_id'] = 3
        write_json(self.run / 'notes/strategy.json', draft)
        with self.assertRaisesRegex(ValueError, 'invalid or duplicate'):
            build_artifacts(self.run)

    def test_recipe_rejects_missing_actions_and_missing_conditions(self):
        self.success()
        draft = read_json(self.run / 'notes/strategy.json')
        original = dict(draft['steps'][1])
        draft['steps'].pop()
        write_json(self.run / 'notes/strategy.json', draft)
        with self.assertRaisesRegex(ValueError, 'omits'):
            build_artifacts(self.run)
        draft['steps'].append(original)
        draft['steps'][0]['verify'] = ''
        write_json(self.run / 'notes/strategy.json', draft)
        with self.assertRaisesRegex(ValueError, 'requires verify'):
            build_artifacts(self.run)

    def test_publication_is_retryable_and_includes_image_evidence(self):
        self.success()
        receipt = publish_run(self.run)
        self.assertEqual(publish_run(self.run), receipt)
        self.assertTrue(any(name.endswith('final/global.png') for name in receipt['files']))
        target = self.memory / 'task-specific' / (self.configuration['cell_tag'] + '_recipe.jsonl')
        target.unlink()
        publish_run(self.run)
        self.assertTrue(target.exists())

    def test_concurrent_publications_keep_a_matching_pair(self):
        self.success()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _:publish_run(self.run),range(2)))
        self.assertEqual(results[0],results[1])

    def test_snapshot_is_independent_and_detects_changes(self):
        self.success()
        publish_run(self.run)
        frozen = self.root / 'snapshot-v1'
        manifest = freeze_memory(self.memory, frozen)
        self.assertEqual(verify_snapshot(frozen)['id'],manifest['id'])
        (self.memory / 'task-specific' / (self.configuration['cell_tag'] + '.json')).write_text('{}')
        self.assertEqual(verify_snapshot(frozen)['id'],manifest['id'])
        (frozen / 'extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'file set'):
            verify_snapshot(frozen)

    def test_freeze_requires_success_and_complete_receipts(self):
        with self.assertRaisesRegex(ValueError,'No published'):
            freeze_memory(self.memory,self.root/'frozen')
        self.success()
        publish_run(self.run)
        pair = self.memory / 'task-specific' / (self.configuration['cell_tag'] + '.json')
        pair.write_text('{}')
        with self.assertRaisesRegex(ValueError,'receipt'):
            freeze_memory(self.memory,self.root/'frozen')

    def test_evaluation_and_baseline_match_the_frozen_configuration(self):
        self.success()
        publish_run(self.run)
        frozen = self.root / 'frozen'
        freeze_memory(self.memory,frozen)
        args = SimpleNamespace(protocol='evaluate',snapshot=frozen,output_dir=self.root/'eval')
        args.output_dir.mkdir()
        selected,manifest = prepare_memory(args,self.configuration)
        self.assertEqual(selected,frozen)
        args.protocol = 'zero-shot'
        empty,baseline_manifest = prepare_memory(args,self.configuration)
        self.assertFalse(any(empty.iterdir()))
        self.assertEqual(manifest['id'],baseline_manifest['id'])
        changed = dict(self.configuration,model='different-model')
        with self.assertRaisesRegex(ValueError,'model'):
            prepare_memory(args,changed)

    def test_file_tools_isolate_notes_evidence_and_frozen_memory(self):
        memory = RunMemory(self.memory,self.run,cell_tag='current')
        self.assertEqual(memory.authorize_write(self.run/'notes/draft.json'),self.run/'notes/draft.json')
        with self.assertRaises(PermissionError):
            memory.authorize_write(self.run/'result.json')
        with self.assertRaises(PermissionError):
            memory.authorize_read(self.root/'other-attempt/result.json')
        allowed = self.memory/'_internal/inbox/current/attempts.json'
        self.assertEqual(memory.authorize_write(allowed),allowed)
        with self.assertRaises(PermissionError):
            memory.authorize_read(self.memory/'_internal/inbox/other/draft.md')
        readonly = RunMemory(self.memory,self.run)
        with self.assertRaises(PermissionError):
            readonly.authorize_write(self.memory/'task-specific/recipe.jsonl')

    def test_interrupted_native_entry_persists_result_and_final_images(self):
        import harness_vla
        planner = SimpleNamespace(solve=lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
        output = self.root/'interrupted-run'
        argv=['harness_vla.py','--model','openai-chat:test','--ur5-repo',str(ROOT),
              '--config',str(ROOT/'config/ur5.example.toml'),'--output-dir',str(output)]
        with patch.object(sys,'argv',argv), patch('harness_vla.load_dotenv'), \
             patch('rpent.planner.base.build_planner',return_value=planner), \
             patch('policy_rollout.UR5Robot',return_value=FakeRobot()), \
             patch('harness_vla.subprocess.run'):
            self.assertEqual(harness_vla.main(),130)
        result=read_json(output/'result.json')
        self.assertTrue(result['interrupted'])
        self.assertIsNone(result['physical_success'])
        self.assertEqual(set(result['final_evidence']),{'global.png','wrist.png','state.json'})

    def test_hardware_initialization_failure_is_durable(self):
        import harness_vla
        output = self.root/'failed-run'
        argv=['harness_vla.py','--model','openai-chat:test','--ur5-repo',str(ROOT),
              '--config',str(ROOT/'config/ur5.example.toml'),'--output-dir',str(output)]
        with patch.object(sys,'argv',argv), patch('harness_vla.load_dotenv'), \
             patch('rpent.planner.base.build_planner'), \
             patch('policy_rollout.UR5Robot',side_effect=RuntimeError('RTDE occupied')), \
             patch('harness_vla.subprocess.run'):
            self.assertEqual(harness_vla.main(),1)
        self.assertIn('RTDE occupied',read_json(output/'result.json')['error'])

    def test_distillation_is_file_only_and_publishes_an_evidence_backed_pair(self):
        from harness_workflow import distill
        self.success()
        strategy = read_json(self.run/'notes/strategy.json')
        (self.run/'notes/strategy.json').unlink()

        def solve(**kwargs):
            toolkit = kwargs['toolkit']
            self.assertFalse(any(tool.name in {'pi05_act','move_delta','set_gripper'} for tool in toolkit.list_tools()))
            result = toolkit.execute_tool('write_text_file',dict(path=str(self.run/'notes/strategy.json'),
                                          content=json.dumps(strategy)))
            self.assertFalse(result.is_error,result.data)
            return SimpleNamespace(finish_result={'status':'success'},stats={'turns_used':1},
                                   error=None,messages=[])

        with patch('rpent.planner.base.build_planner',return_value=SimpleNamespace(solve=solve)), \
             patch('harness_workflow.load_dotenv'):
            distill(self.run)
        publish_run(self.run)
        frozen = self.root/'distilled-snapshot'
        manifest = freeze_memory(self.memory,frozen)
        self.assertTrue(any(name.endswith('_recipe.jsonl') for name in manifest['files']))

    def test_exploration_retains_failed_attempt_context_for_the_next_attempt(self):
        import harness_workflow
        args = SimpleNamespace(output_dir=self.root/'session',memory_dir=self.memory,
                               reference_description='Both objects outside the box',scene_id='reference-0',attempts=0)
        created = []

        def child(arguments):
            directory = Path(arguments[arguments.index('--output-dir')+1])
            directory.mkdir()
            (directory/'notes').mkdir()
            write_json(directory/'notes/failure.json',{'observation':'empty grasp'})
            write_json(directory/'result.json',{'error':'gripper communication fault','finish':None})
            created.append(directory)
            return 1

        with patch('builtins.input',side_effect=['ready','q']), \
             patch('harness_workflow.child_run',side_effect=child), \
             patch('harness_workflow.operator_review',return_value={'physical_success':False,'autonomous_success':False,'outcome':'hardware-error'}):
            self.assertEqual(harness_workflow.explore(args,[]),0)
        session = read_json(args.output_dir/'session.json')
        inbox = self.memory/'_internal/inbox'/session['cell_tag']
        history=read_json(inbox/'attempts.json')
        self.assertIn('empty grasp',str(history))
        self.assertEqual(len(session['attempts']),1)
        self.assertFalse((self.memory/'task-specific').exists())

    def test_normal_exploration_entry_reaches_review_publish_and_freeze(self):
        import harness_vla
        output=self.root/'native-exploration'
        robot=FakeRobot()

        def solve(**kwargs):
            toolkit=kwargs['toolkit']
            result=toolkit.execute_tool('move_delta',dict(dx_mm=2.,dy_mm=0.,dz_mm=0.))
            self.assertFalse(result.is_error,result.data)
            return SimpleNamespace(finish_result={'status':'success'},stats={'turns_used':1},
                                   error=None,messages=[])

        argv=['harness_vla.py','--model','openai-chat:test','--protocol','explore',
              '--mode','record','--execute','--i-understand','--ur5-repo',str(ROOT),
              '--config',str(ROOT/'config/ur5.example.toml'),'--output-dir',str(output),
              '--memory-dir',str(self.memory)]
        identity={'action_horizon':50,'num_steps':10,'checkpoint_sha256':'a'*64}
        with patch.object(sys,'argv',argv), patch('harness_vla.load_dotenv'), \
             patch('harness_protocol.policy_identity',return_value=identity), \
             patch('rpent.planner.base.build_planner',return_value=SimpleNamespace(solve=solve)), \
             patch('policy_rollout.UR5Robot',return_value=robot), \
             patch('policy_rollout.prepare_execution'), patch('harness_vla.subprocess.run'):
            self.assertEqual(harness_vla.main(),0)
        self.assertIsNone(read_json(output/'result.json')['physical_success'])
        configuration=read_json(output/'configuration.json')
        self.assertTrue((self.memory/'_internal/inbox'/configuration['cell_tag']).is_dir())
        review_run(output,outcome='success',notes='Fake hardware integration test',
                   conditions=dict.fromkeys(SUCCESS_CONDITIONS,True))
        write_json(output/'notes/strategy.json',dict(applicability='Test scene',
                   steps=[dict(call_id=1,when='Target observed',verify='Target reached',
                               on_failure='Reobserve',rebind='Use current target and TCP')]))
        publish_run(output)
        snapshot=freeze_memory(self.memory,self.root/'native-snapshot')
        self.assertEqual(snapshot['contract']['checkpoint_sha256'],'a'*64)

    def test_policy_identity_only_reads_the_server_handshake(self):
        from harness_protocol import policy_identity
        identity = {'action_horizon':50,'num_steps':10,'checkpoint_sha256':'a'*64}
        with patch('websockets.sync.client.connect') as connect, \
             patch('openpi_client.msgpack_numpy.unpackb',return_value=identity):
            self.assertEqual(policy_identity('127.0.0.1',8000,3),identity)
            connect.return_value.__enter__.return_value.send.assert_not_called()
        with patch('websockets.sync.client.connect'), \
             patch('openpi_client.msgpack_numpy.unpackb',return_value=dict(identity,num_steps=9)):
            with self.assertRaisesRegex(ValueError,'full-policy'):
                policy_identity('127.0.0.1',8000,3)

    def test_configuration_mismatch_prevents_hardware_opening_and_is_recorded(self):
        import harness_vla
        self.success()
        publish_run(self.run)
        frozen=self.root/'mismatched-snapshot'
        freeze_memory(self.memory,frozen)
        output=self.root/'mismatched-evaluation'
        argv=['harness_vla.py','--model','different-model','--protocol','evaluate',
              '--snapshot',str(frozen),'--ur5-repo',str(ROOT),
              '--config',str(ROOT/'config/ur5.example.toml'),'--output-dir',str(output)]
        identity={'action_horizon':50,'num_steps':10,'checkpoint_sha256':'a'*64}
        with patch.object(sys,'argv',argv), patch('harness_vla.load_dotenv'), \
             patch('harness_protocol.policy_identity',return_value=identity), \
             patch('policy_rollout.UR5Robot') as robot, patch('harness_vla.subprocess.run'):
            self.assertEqual(harness_vla.main(),1)
            robot.assert_not_called()
        self.assertIn('frozen configuration',read_json(output/'result.json')['error'])

    def test_summary_keeps_unreviewed_and_hardware_failure_trials_visible(self):
        from harness_workflow import summarize
        review_run(self.run,outcome='hardware-error',notes='serial fault',conditions={})
        report=summarize(self.root)
        self.assertEqual(report['groups'][0]['hardware_errors'],1)
        self.assertEqual(report['groups'][0]['success_rate'],0)
        self.assertEqual(report['runs'][0]['vla_calls'],2)


if __name__ == '__main__':
    unittest.main()
