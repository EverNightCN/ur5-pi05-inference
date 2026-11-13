"""One acceptance rule for the complete, audited pi0.5 checkpoint."""
import hashlib
import json
from pathlib import Path

ARTIFACTS = {"model.safetensors", "config.json", "norm_stats.json",
             "paligemma_tokenizer.model", "source_model_spec.json"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_checkpoint(directory):
    directory = Path(directory)
    audit = json.loads((directory / "conversion_audit.json").read_text())
    expected_paths = set()
    for suffix in ("", "_1"):
        expected_paths.update(f"PaliGemma/llm/layers/attn/{name}{suffix}/w"
                              for name in ("q_einsum", "kv_einsum", "attn_vec_einsum"))
        expected_paths.update(f"PaliGemma/llm/layers/mlp{suffix}/{name}"
                              for name in ("gating_einsum", "linear"))
    groups = audit.get("merge_groups", [])
    if (audit.get("strict_load") is not True or len(groups) != 10
            or {group["path"] for group in groups} != expected_paths):
        raise ValueError("Checkpoint must contain all ten audited LoRA groups and a strict load")
    for group in groups:
        if "attn_vec_einsum" in group["path"] and group.get("sum_b_heads") is not True:
            raise ValueError("Checkpoint has the obsolete per-head output LoRA merge")
    checksums = audit["files_sha256"]
    if set(checksums) != ARTIFACTS:
        raise ValueError("Audit must hash exactly the five checkpoint artifacts, without self-reference")
    for name in sorted(ARTIFACTS):
        if sha256(directory / name) != checksums[name]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    config = json.loads((directory / "config.json").read_text())
    expected = {"pi05": True, "action_dim": 32, "action_horizon": 50,
                "max_token_len": 200, "num_steps": 10,
                "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Checkpoint config {key} must be {value!r}, got {config.get(key)!r}")
    return audit
