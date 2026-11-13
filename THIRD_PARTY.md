# Third-party software

| Component | Source | License | Version / provenance |
|---|---|---|---|
| UR5 hardware support | [HDU-EILab/ur5-teleop](https://github.com/HDU-EILab/ur5-teleop) | MIT | Local deployed hardware support snapshot, with existing inference adaptations |
| RTDE Python client | [UniversalRobots/RTDE_Python_Client_Library](https://github.com/UniversalRobots/RTDE_Python_Client_Library) | BSD-3-Clause | Vendored source with `RTDE_LICENSE` |
| OpenPI runtime | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0 | Tested runtime source archive SHA-256 `742591545f8e860e03335f7e72726da9ad9f64e618f39694dbfda617b1a0234a` |
| RPent | [RLinf/RPent](https://github.com/RLinf/RPent/tree/ed6b9223fa8e9854964d067c94d7df3ded90e75a) | Apache-2.0 | `ed6b9223fa8e9854964d067c94d7df3ded90e75a`, downloaded by `setup_harness.sh` |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154) | Apache-2.0 | Rollout organization reference |

## OpenPI source snapshot

`vendor/openpi` preserves the source archive used by the tested AGX runtime. Its per-file manifest is `SOURCE.sha256.json`. The archive's environment-specific `pyproject.toml` is omitted because this repository installs its own pinned dependency files. The snapshot already contains adaptations relative to upstream OpenPI, including the deployed PyTorch/Transformers implementation. It differs from the training metadata's upstream commit `a560f4dd8205b8423ecd4c8a0fabb5f54140b8a0`, so that commit is used only as training provenance.

`inference.patch` applies the additional Jetson SDPA, fixed-step sampling, tensor allocation, preprocessing and runtime annotation changes. `tools/prepare_runtime.py` verifies the source snapshot, creates a fresh runtime copy, applies the patch, and installs the snapshot's Transformers replacement files into the dedicated environment.

The Apache-2.0 license is retained at `vendor/openpi/LICENSE`. Preserve that license and this provenance notice when redistributing the snapshot or its modified runtime.

## Software packages and model artifacts

Torch/Triton wheels and NVIDIA cuDSS are fetched from their public distribution URLs. Their original licenses govern their use. RPent is fetched as a fixed, checksummed source archive and uses its own Apache-2.0 license. Python packages retain their upstream licenses.

Model checkpoint distribution and use are governed by the corresponding model and dataset terms. Users supply checkpoint artifacts separately when deploying this repository.
