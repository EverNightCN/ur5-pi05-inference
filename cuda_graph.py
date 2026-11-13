"""Capture the complete fixed-shape model; preprocessing and transfers stay outside."""
import jax
import torch


class CudaGraphSampler:
    def __init__(self, model):
        self.model = model
        self.graph = None

    @torch.no_grad()
    def __call__(self, device, observation, *, noise=None, num_steps=10):
        if num_steps != 10:
            raise ValueError("The deployed graph requires the complete 10-step inference configuration")
        if noise is None:
            noise = self.model.sample_noise((observation.state.shape[0], 50, 32), device)
        if self.graph is None:
            self._capture(device, observation, noise)
        else:
            if jax.tree.structure(observation) != jax.tree.structure(self.observation):
                raise ValueError("Observation structure differs from captured graph")
            def copy(dst, src):
                if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                    raise ValueError("Observation shape, dtype or device differs from captured graph")
                dst.copy_(src)
            jax.tree.map(copy, self.observation, observation)
            if noise.shape != self.noise.shape or noise.dtype != self.noise.dtype:
                raise ValueError("Noise shape or dtype differs from captured graph")
            self.noise.copy_(noise)
        self.graph.replay()
        return self.output.clone()

    def _capture(self, device, observation, noise):
        self.observation = jax.tree.map(lambda x: x.clone(), observation)
        self.noise = noise.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model.sample_actions(device, self.observation, noise=self.noise, num_steps=10)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.model.sample_actions(device, self.observation, noise=self.noise, num_steps=10)
