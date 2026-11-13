"""Strict, audited conversion of the UR5 pi05 LoRA checkpoint; CPU only."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from checkpoint import ARTIFACTS, sha256 as digest


def merge_lora(params, configs):
    """Merge the exact Attention/FeedForward formulas in openpi.models.lora."""
    audit = []
    layers = params['PaliGemma']['llm']['layers']

    def combine(node, base, a_key, b_key, rank, scale, path, *, sum_b_heads=False):
        if not all(k in node for k in (base, a_key, b_key)):
            raise ValueError(f'Incomplete LoRA group: {path}')
        weight, a, b = (np.asarray(node[k], dtype=np.float32) for k in (base, a_key, b_key))
        if a.shape[-1] != rank or b.shape[-2] != rank:
            raise ValueError(f'LoRA rank mismatch: {path}: {a.shape}, {b.shape}, expected {rank}')
        if a.shape[:-2] != weight.shape[:-2] or b.shape[:-2] != weight.shape[:-2]:
            raise ValueError(f'LoRA batch shape mismatch: {path}')
        # The output projection's LoRA equations reduce N independently:
        # BTNH,NHL->BTL followed by BTL,NLD->BTD. With scanned layers,
        # preserve axis 0 and sum only B's head axis 1 before merging.
        if sum_b_heads:
            if a.ndim != 4 or b.ndim != 4:
                raise ValueError(f'Expected [layer, head, dim, rank] output LoRA: {path}')
            b = b.sum(axis=1, keepdims=True)
        delta = np.matmul(a, b) * scale
        if delta.shape != weight.shape:
            raise ValueError(f'LoRA output shape mismatch: {path}: {delta.shape} != {weight.shape}')
        if not np.isfinite(delta).all() or not np.isfinite(weight).all():
            raise ValueError(f'Non-finite weight: {path}')
        audit.append({'path': path, 'shape': list(weight.shape), 'rank': rank,
                      'scale': float(scale), 'sum_b_heads': sum_b_heads, 'delta_abs_max': float(np.max(np.abs(delta))),
                      'delta_l2': float(np.linalg.norm(delta.ravel()))})
        node[base] = weight + delta
        del node[a_key], node[b_key]

    for suffix, config in configs.items():
        attn = config.lora_configs['attn']
        ffn = config.lora_configs['ffn']
        if attn.axes != (-2, -1):
            raise ValueError(f'Unsupported attention LoRA axes: {attn.axes}')
        for name in ('q_einsum', 'kv_einsum', 'attn_vec_einsum'):
            module = name + suffix
            combine(layers['attn'][module], 'w', 'lora_a', 'lora_b', attn.rank,
                    attn.scaling_value, f'PaliGemma/llm/layers/attn/{module}/w',
                    sum_b_heads=name == 'attn_vec_einsum')
        for name in ('gating_einsum', 'linear'):
            # FeedForward._dot adds A@B directly, without LoRAConfig scaling.
            combine(layers['mlp' + suffix], name, name + '_lora_a', name + '_lora_b',
                    ffn.rank, 1.0, f'PaliGemma/llm/layers/mlp{suffix}/{name}')

    def reject_remaining(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                if 'lora' in key.lower():
                    raise ValueError('Unconsumed LoRA key: ' + '/'.join((*path, key)))
                reject_remaining(value, (*path, key))
    reject_remaining(params)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {args.output}')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    sys.path.insert(0, str(args.openpi / 'src'))
    from flax.nnx import traversals
    import safetensors.torch
    import torch
    from openpi.models import gemma, model as model_lib
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    source_spec = json.loads((args.checkpoint / 'model_spec.json').read_text())
    if source_spec['model'] != 'pi05':
        raise ValueError('Expected pi05 checkpoint')
    configs = {'': gemma.get_config(source_spec['paligemma_variant']),
               '_1': gemma.get_config(source_spec['action_expert_variant'])}
    print('Restoring full JAX checkpoint on CPU', flush=True)
    params = model_lib.restore_params(str(args.checkpoint / 'params'), restore_type=np.ndarray, dtype='float32')
    audit = merge_lora(params, configs)
    print(f'Merged {len(audit)} stacked LoRA groups (attention and FFN)', flush=True)
    cfg = Pi0Config(pi05=True, action_dim=source_spec['action_dim'],
                    action_horizon=source_spec['action_horizon'], max_token_len=source_spec['max_token_len'],
                    pytorch_compile_mode=None)
    converter_path = args.openpi / 'examples/convert_jax_model_to_pytorch.py'
    spec = importlib.util.spec_from_file_location('openpi_converter', converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    print('Constructing PyTorch model', flush=True)
    from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        model = PI0Pytorch(cfg)
    # no_init_weights also skips HF post-init weight tying. Restore the declared alias.
    model.paligemma_with_expert.paligemma.tie_weights()
    flat = traversals.flatten_mapping(params['PaliGemma'], sep='/')
    vlm, expert = converter.slice_paligemma_state_dict(flat, model.paligemma_with_expert.paligemma.config)
    # Upstream dispatches adarms by directory name rather than its pi05 argument.
    # An explicit pi05 marker ensures dispatch is independent of the input path.
    expert = converter.slice_gemma_state_dict(expert, gemma.get_config('gemma_300m'),
                                              num_expert=1, checkpoint_dir='pi05', pi05=True)
    weights = {**vlm, **expert}
    for name in ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'):
        weights[name + '.weight'] = torch.from_numpy(params[name]['kernel']).T
        weights[name + '.bias'] = torch.from_numpy(params[name]['bias'])
    embed = 'paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    tied_head = 'paligemma_with_expert.paligemma.lm_head.weight'
    unused_head = 'paligemma_with_expert.gemma_expert.lm_head.weight'
    named = dict(model.named_parameters(remove_duplicate=False))
    if named[tied_head] is not named[embed]:
        raise ValueError('Expected tied PaliGemma embedding/head')
    weights[tied_head] = weights[embed]
    # Action inference calls gemma_expert.model, never its vocabulary output head.
    # Initialize this unused compatibility parameter deterministically.
    weights[unused_head] = torch.zeros_like(named[unused_head])
    print('Loading converted tensors with strict=True', flush=True)
    model.load_state_dict(weights, strict=True)
    args.output.mkdir(parents=True)
    model_path = args.output / 'model.safetensors'
    safetensors.torch.save_model(model, str(model_path))
    for name in ('norm_stats.json', 'paligemma_tokenizer.model'):
        shutil.copy2(args.checkpoint / name, args.output / name)
    shutil.copy2(args.checkpoint / 'model_spec.json', args.output / 'source_model_spec.json')
    config = {'pi05': True, 'action_dim': cfg.action_dim, 'action_horizon': cfg.action_horizon,
              'max_token_len': cfg.max_token_len, 'paligemma_variant': cfg.paligemma_variant,
              'action_expert_variant': cfg.action_expert_variant,
              'precision': 'bfloat16 with official float32 parameters', 'num_steps': 10}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = {'source_checkpoint': str(args.checkpoint), 'source_spec': source_spec,
              'merge_groups': audit, 'strict_load': True,
              'tied_alias': {tied_head: embed}, 'unused_zero_initialized_parameter': unused_head,
              'converter_sha256': digest(converter_path),
              'merge_implementation_sha256': digest(Path(__file__))}
    pending = args.output / '.conversion_audit.pending.json'
    pending.write_text(json.dumps(report, indent=2) + '\n')
    # Release both restored FP32 arrays and the model before reading the large file
    # again for its digest. Inference validation may run concurrently on this host.
    del model, named, weights, vlm, expert, flat, params
    gc.collect()
    report['files_sha256'] = {name: digest(args.output / name) for name in sorted(ARTIFACTS)}
    (args.output / 'conversion_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    pending.unlink()
    print('Conversion complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
