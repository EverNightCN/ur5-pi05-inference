import json
from pathlib import Path
import tempfile
import unittest

from checkpoint import ARTIFACTS, sha256, validate_checkpoint


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ARTIFACTS:
            (self.root / name).write_bytes(b"test artifact")
        config = {"pi05": True, "action_dim": 32, "action_horizon": 50, "max_token_len": 200,
                  "num_steps": 10, "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"}
        (self.root / "config.json").write_text(json.dumps(config))
        groups = []
        for suffix in ("", "_1"):
            for name in ("q_einsum", "kv_einsum", "attn_vec_einsum"):
                groups.append({"path": f"PaliGemma/llm/layers/attn/{name}{suffix}/w",
                               "sum_b_heads": name == "attn_vec_einsum"})
            for name in ("gating_einsum", "linear"):
                groups.append({"path": f"PaliGemma/llm/layers/mlp{suffix}/{name}"})
        self.audit = {"strict_load": True, "merge_groups": groups,
                      "files_sha256": {name: sha256(self.root / name) for name in ARTIFACTS}}
        self.save()

    def save(self):
        (self.root / "conversion_audit.json").write_text(json.dumps(self.audit))

    def test_valid_artifact_manifest(self):
        validate_checkpoint(self.root)

    def test_rejects_known_incorrect_merge(self):
        for group in self.audit["merge_groups"]:
            group.pop("sum_b_heads", None)
        self.save()
        with self.assertRaisesRegex(ValueError, "obsolete per-head"):
            validate_checkpoint(self.root)

    def test_rejects_modified_normalization(self):
        (self.root / "norm_stats.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch: norm_stats"):
            validate_checkpoint(self.root)

    def test_rejects_self_referential_hash(self):
        self.audit["files_sha256"]["conversion_audit.json"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "without self-reference"):
            validate_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
