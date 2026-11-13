"""Check CUDA BF16, Triton compilation and runtime imports without model weights."""
import json
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the dedicated inference environment')
    if torch.cuda.get_device_capability() != (8, 7):
        raise RuntimeError('This runtime targets Jetson AGX Orin SM87')
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all().item()
    compiled = torch.compile(lambda value: value.sin() + value.cos(), fullgraph=True)
    assert torch.isfinite(compiled(torch.randn(1024, device='cuda'))).all().item()
    torch.cuda.synchronize()
    import runtime
    print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                          device=torch.cuda.get_device_name(), bf16=True,
                          triton_compile=True, runtime_imported=True)))


if __name__ == '__main__':
    main()
