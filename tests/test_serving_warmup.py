"""Verify serving warmup order without importing the GPU runtime."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


class ServingWarmupTest(unittest.TestCase):
    def run_server(self, extra):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / 'source_model_spec.json').write_text(json.dumps({'prompt': 'checkpoint task'}))
            events = []
            policy = SimpleNamespace(metadata={}, infer=lambda obs: events.append(('infer', obs['prompt'])))
            server = MagicMock()
            def create_server(*args, **kwargs):
                events.append(('serve',))
                return server
            cuda = SimpleNamespace(synchronize=MagicMock())
            modules = {
                'runtime': SimpleNamespace(load_policy=MagicMock(return_value=policy),
                                           read_observation=lambda _: {'prompt': 'default task'}),
                'torch': SimpleNamespace(set_num_threads=MagicMock(), cuda=cuda),
                'openpi': SimpleNamespace(),
                'openpi.serving': SimpleNamespace(),
                'openpi.serving.websocket_policy_server': SimpleNamespace(WebsocketPolicyServer=create_server),
            }
            arguments = ['serve.py', '--checkpoint', str(checkpoint), '--warmup-sample', 'input.npz', *extra]
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', arguments), \
                    patch('logging.basicConfig'):
                runpy.run_path(str(ROOT / 'serve.py'), run_name='__main__')
            server.serve_forever.assert_called_once()
            return events, cuda.synchronize.call_count

    def test_three_checkpoint_prompt_warmups_precede_server_readiness(self):
        events, count = self.run_server(['--warmup-runs', '3', '--warmup-prompt-from-checkpoint'])
        self.assertEqual(events, [('infer', 'checkpoint task')] * 3 + [('serve',)])
        self.assertEqual(count, 3)

    def test_default_invocation_preserves_single_default_prompt_warmup(self):
        events, count = self.run_server([])
        self.assertEqual(events, [('infer', 'default task'), ('serve',)])
        self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
