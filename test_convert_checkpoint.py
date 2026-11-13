"""Regression tests against the actual Gemma attention einsum equations."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
from convert_checkpoint import merge_lora


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(23)
        self.configs = {}
        self.layers = {'attn': {}}
        for suffix in ('', '_1'):
            self.configs[suffix] = SimpleNamespace(lora_configs={
                'attn': SimpleNamespace(rank=2, scaling_value=0.25, axes=(-2, -1)),
                'ffn': SimpleNamespace(rank=2, scaling_value=0.25)})
            # Axes: scanned layer, optional P/K/N, input dim, output dim.
            for name, shape in (('q_einsum', (2, 3, 4, 5)),
                                ('kv_einsum', (2, 2, 2, 4, 5)),
                                ('attn_vec_einsum', (2, 3, 5, 4))):
                self.layers['attn'][name + suffix] = self.group('w', 'lora_a', 'lora_b', shape)
            self.layers['mlp' + suffix] = {}
            for name, shape in (('gating_einsum', (2, 2, 4, 7)), ('linear', (2, 7, 4))):
                self.layers['mlp' + suffix].update(
                    self.group(name, name + '_lora_a', name + '_lora_b', shape))
        self.params = {'PaliGemma': {'llm': {'layers': self.layers}}}

    def group(self, w, a, b, shape):
        return {w: self.rng.normal(size=shape).astype('float32'),
                a: self.rng.normal(size=(*shape[:-1], 2)).astype('float32'),
                b: self.rng.normal(size=(*shape[:-2], 2, shape[-1])).astype('float32')}

    def test_actual_attention_einsum_equivalence(self):
        before = copy.deepcopy(self.layers)
        audit = merge_lora(self.params, self.configs)
        self.assertEqual(len(audit), 10)
        # Equations from gemma.Attention and lora.Einsum._make_lora_eqns.
        # NumPy uses P instead of the JAX equation's literal axis label '2'.
        equations = {
            'q_einsum': ('BTD,NDH->BTNH', 'BTD,NDL->BTNL', 'BTNL,NLH->BTNH', (2, 3, 4)),
            'kv_einsum': ('BSD,PKDH->PBSKH', 'BSD,PKDL->PBSKL', 'PBSKL,PKLH->PBSKH', (2, 3, 4)),
            'attn_vec_einsum': ('BTNH,NHD->BTD', 'BTNH,NHL->BTL', 'BTL,NLD->BTD', (2, 3, 3, 5)),
        }
        for suffix in ('', '_1'):
            for name, (base_eq, a_eq, b_eq, shape) in equations.items():
                old = before['attn'][name + suffix]
                for layer in range(2):
                    x = self.rng.normal(size=shape).astype('float32')
                    base = np.einsum(base_eq, x, old['w'][layer])
                    adapter = np.einsum(b_eq, np.einsum(a_eq, x, old['lora_a'][layer]), old['lora_b'][layer])
                    expected = base + adapter * 0.25
                    actual = np.einsum(base_eq, x, self.layers['attn'][name + suffix]['w'][layer])
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                    if name == 'attn_vec_einsum':
                        wrong = old['w'][layer] + old['lora_a'][layer] @ old['lora_b'][layer] * 0.25
                        self.assertGreater(np.max(np.abs(np.einsum(base_eq, x, wrong) - expected)), 0.01)

    def test_ffn_dot_formula_has_no_scaling(self):
        before = copy.deepcopy(self.layers)
        merge_lora(self.params, self.configs)
        for suffix in ('', '_1'):
            for name in ('gating_einsum', 'linear'):
                old = before['mlp' + suffix]
                for layer in range(2):
                    indices = ((layer, 0), (layer, 1)) if name == 'gating_einsum' else ((layer,),)
                    for index in indices:
                        w, a, b = (old[k][index] for k in (name, name + '_lora_a', name + '_lora_b'))
                        x = self.rng.normal(size=(2, 3, w.shape[0])).astype('float32')
                        expected = np.dot(x, w) + np.dot(np.dot(x, a), b)
                        actual = np.dot(x, self.layers['mlp' + suffix][name][index])
                        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_unknown_lora_is_rejected(self):
        self.params['unrecognized_lora'] = np.zeros(1)
        with self.assertRaisesRegex(ValueError, 'Unconsumed LoRA'):
            merge_lora(self.params, self.configs)

    def test_missing_pair_is_rejected(self):
        del self.layers['mlp']['linear_lora_b']
        with self.assertRaisesRegex(ValueError, 'Incomplete LoRA'):
            merge_lora(self.params, self.configs)

    def test_wrong_rank_is_rejected(self):
        self.layers['attn']['q_einsum']['lora_a'] = np.zeros((2, 3, 4, 7))
        with self.assertRaisesRegex(ValueError, 'rank mismatch'):
            merge_lora(self.params, self.configs)

    def test_nonfinite_is_rejected(self):
        self.layers['attn']['q_einsum']['w'][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            merge_lora(self.params, self.configs)


if __name__ == '__main__':
    unittest.main()
