#!/usr/bin/env python3
"""
Device detection and setup for ClarityOCR.
Supports CUDA (NVIDIA) and CPU fallback.
"""

from typing import Literal, Optional

DeviceType = Literal["cuda", "cpu"]


def get_device_type() -> DeviceType:
    """Detect the best available runtime device."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_name() -> str:
    """Return a human-readable device name."""
    try:
        import torch
    except ImportError:
        return "CPU (torch not installed)"

    if get_device_type() == "cuda":
        try:
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA (unknown model)"
    return "CPU"


def configure_device() -> Optional[str]:
    """Apply runtime optimizations for the detected device."""
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"

    if get_device_type() == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return "CUDA configured: benchmark + TF32 enabled"
    return "Using CPU"


def get_memory_info() -> dict:
    """Return memory stats for the active runtime."""
    if get_device_type() == "cuda":
        try:
            import torch

            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1024**3
            used = torch.cuda.memory_reserved() / 1024**3
            return {
                "total": total,
                "used": used,
                "free": total - used,
                "type": "cuda",
            }
        except Exception:
            return {"total": 0.0, "used": 0.0, "free": 0.0, "type": "cuda"}

    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,
            "used": mem.used / 1024**3,
            "free": mem.available / 1024**3,
            "type": "cpu",
        }
    except Exception:
        return {"total": 0.0, "used": 0.0, "free": 0.0, "type": "cpu"}


def cleanup_device() -> None:
    """Clear runtime caches."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and get_device_type() == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    import gc

    gc.collect()


def is_gpu_available() -> bool:
    """Return True only when CUDA is available."""
    return get_device_type() == "cuda"


def get_torch_device() -> str:
    """Return torch-compatible device string."""
    return get_device_type()


if __name__ == "__main__":
    print("=== ClarityOCR Device Detection ===")
    print(f"Device Type: {get_device_type()}")
    print(f"Device Name: {get_device_name()}")
    print(f"GPU Available: {is_gpu_available()}")

    config = configure_device()
    if config:
        print(config)

    mem = get_memory_info()
    print(f"Memory total: {mem['total']:.2f} GB")
    print(f"Memory used:  {mem['used']:.2f} GB")
    print(f"Memory free:  {mem['free']:.2f} GB")
