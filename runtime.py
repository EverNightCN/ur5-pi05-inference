"""Local UR5 pi0.5 inference; no robot commands are issued here."""
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from checkpoint import validate_checkpoint

# openpi's logging utilities use this Python 3.11 name on Jetson Python 3.10.
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

import numpy as np
import torch
import sentencepiece
import safetensors.torch
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
from openpi.policies.policy import Policy
from openpi_client import image_tools
from openpi.shared import normalize
# Initialize openpi/JAX before modeling_utils; the reverse order crashes this pinned ARM stack during import.
from transformers.modeling_utils import no_init_weights

PROMPT = ("Put the red cylinder and the blue cube into the storage box. "
          "Leave all other objects outside the box.")


class LocalTokenizer(tokenizer.PaligemmaTokenizer):
    def __init__(self, path):
        self._max_len = 200
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=Path(path).read_bytes())


class ParallelResizeImages:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pi05-images")

    def __call__(self, data):
        keys = list(data["image"])
        images = self.pool.map(self.resize, (data["image"][key] for key in keys))
        data["image"] = dict(zip(keys, images, strict=True))
        return data

    @staticmethod
    def resize(image):
        return image_tools.resize_with_pad(image, 224, 224)


class CheckedPolicy(Policy):
    def infer(self, obs, *, noise=None):
        validate_observation(obs)
        if noise is not None and (not isinstance(noise, np.ndarray)
                or noise.shape not in ((50, 32), (1, 50, 32))
                or noise.dtype != np.float32 or not np.isfinite(noise).all()):
            raise ValueError("noise must be finite float32 with shape (50,32) or (1,50,32)")
        return super().infer(obs, noise=noise)


def set_attention_backend(model, backend):
    if backend not in ("eager", "sdpa"):
        raise ValueError(f"Unsupported attention backend: {backend}")
    model.attention_backend = backend
    model.paligemma_with_expert.paligemma.vision_tower.vision_model.config._attn_implementation = backend


def compile_components(model, mode, all_components=False):
    model.denoise_step = torch.compile(model.denoise_step, mode=mode, fullgraph=True)
    if all_components:
        vlm = model.paligemma_with_expert
        vlm.embed_image = torch.compile(vlm.embed_image, mode=mode, fullgraph=True)
        vlm.paligemma.language_model.forward = torch.compile(
            vlm.paligemma.language_model.forward, mode=mode, fullgraph=True)


@torch.no_grad()
def cache_padding_image(model):
    # LiberoInputs always supplies an all-zero, masked third image for this two-camera policy.
    image = torch.full((1, 224, 224, 3), -1.0, device="cuda").permute(0, 3, 1, 2)
    embedding = model.paligemma_with_expert.embed_image(image)
    model.register_buffer("cached_padding_image", embedding, persistent=False)


def load_policy(checkpoint, *, compile_mode=None, attention_backend="eager", cuda_graph=False,
                compile_all=False, cache_padding=False):
    checkpoint = Path(checkpoint)
    audit = validate_checkpoint(checkpoint)
    cfg = Pi0Config(pi05=True, action_horizon=50, max_token_len=200,
                    pytorch_compile_mode=None)
    # Every parameter is supplied by strict safetensors loading; random initialization is wasted work.
    with no_init_weights():
        model = PI0Pytorch(cfg)
    model.paligemma_with_expert.paligemma.tie_weights()
    safetensors.torch.load_model(model, str(checkpoint / "model.safetensors"), strict=True)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    set_attention_backend(model, attention_backend)
    stats = normalize.load(str(checkpoint))
    if compile_mode is not None:
        compile_components(model, compile_mode, compile_all)
    elif compile_all:
        raise ValueError("compile_all requires an explicit compile_mode")
    policy = CheckedPolicy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(PROMPT),
            LiberoInputs(model_type=model_lib.ModelType.PI05),
            transforms.Normalize(stats, use_quantiles=True),
            ParallelResizeImages(),
            transforms.TokenizePrompt(LocalTokenizer(checkpoint / "paligemma_tokenizer.model"), discrete_state_input=True),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[transforms.Unnormalize(stats, use_quantiles=True), LiberoOutputs()],
        sample_kwargs={"num_steps": 10},
        metadata={"model": "pi05", "action_horizon": 50, "num_steps": 10,
                  "checkpoint_sha256": audit["files_sha256"]["model.safetensors"]},
        is_pytorch=True, pytorch_device="cuda",
    )
    if cache_padding:
        cache_padding_image(model)
    if cuda_graph:
        from cuda_graph import CudaGraphSampler
        policy._sample_actions = CudaGraphSampler(model)
    return policy


def read_observation(path):
    with np.load(path, allow_pickle=False) as sample:
        obs = {key: sample[key] for key in (
            "observation/image", "observation/wrist_image", "observation/state")}
    obs["prompt"] = PROMPT
    validate_observation(obs)
    return obs


def validate_observation(obs):
    required = {"observation/image", "observation/wrist_image", "observation/state"}
    if not isinstance(obs, dict) or not required.issubset(obs):
        raise ValueError(f"Observation must contain {sorted(required)}")
    for key in ("observation/image", "observation/wrist_image"):
        if (not isinstance(obs[key], np.ndarray) or obs[key].dtype != np.uint8
                or obs[key].ndim != 3 or obs[key].shape[-1] != 3 or min(obs[key].shape[:2]) == 0):
            raise ValueError(f"{key} must be an HWC RGB uint8 image")
    state = obs["observation/state"]
    if (not isinstance(state, np.ndarray) or state.shape != (13,)
            or not np.issubdtype(state.dtype, np.floating) or not np.isfinite(state).all()):
        raise ValueError("observation/state must contain 13 finite values")
    if "prompt" in obs and not isinstance(obs["prompt"], str):
        raise ValueError("prompt must be a string")
