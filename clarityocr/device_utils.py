#!/usr/bin/env python3
"""
Device Detection and Setup for ClarityOCR
Supports: CUDA (NVIDIA), MPS (Apple Silicon), CPU
"""

import platform
from typing import Literal, Optional

DeviceType = Literal["cuda", "mps", "cpu"]


def get_device_type() -> DeviceType:
    """
    Detect best available device for inference.

    Priority: CUDA > MPS > CPU

    Returns:
        DeviceType: The detected device type
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_device_name() -> str:
    """
    Get human-readable device name.

    Returns:
        str: Device name or description
    """
    try:
        import torch
    except ImportError:
        return "CPU (torch not installed)"

    device_type = get_device_type()

    if device_type == "cuda":
        try:
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA (unknown model)"

    if device_type == "mps":
        # Get system info for Apple chip name
        sys_info = platform.processor()
        return f"MPS (Apple Silicon: {sys_info})"

    return "CPU"


def configure_device() -> Optional[str]:
    """
    Configure optimal settings for the detected device.

    Returns:
        Optional[str]: Configuration message, or None if no config needed
    """
    try:
        import torch
    except ImportError:
        return "⚠️  PyTorch not installed"

    device_type = get_device_type()
    messages = []

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        messages.append("✅ CUDA configured: benchmark + TF32 enabled")

    elif device_type == "mps":
        # MPS doesn't support all CUDA features
        # Enable high-performance operations where available
        try:
            torch.backends.mps.allow_tf32 = True
        except AttributeError:
            pass  # Older PyTorch versions may not have this
        messages.append("✅ MPS configured (Apple Silicon GPU)")

    else:
        messages.append("ℹ️  Using CPU (consider GPU for better performance)")

    return "\n".join(messages) if messages else None


def get_memory_info() -> dict:
    """
    Get memory information for the detected device.

    Returns:
        dict: Memory info with keys: total, used, free (in GB)
    """
    device_type = get_device_type()

    if device_type == "cuda":
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1024**3
            used = torch.cuda.memory_reserved() / 1024**3
            return {
                "total": total,
                "used": used,
                "free": total - used,
                "type": "cuda"
            }
        except Exception:
            return {"total": 0, "used": 0, "free": 0, "type": "cuda"}

    elif device_type == "mps":
        # MPS doesn't expose detailed memory stats like CUDA
        # Return available system memory approximation
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,
            "used": mem.used / 1024**3,
            "free": mem.available / 1024**3,
            "type": "mps"
        }

    else:
        # CPU - return system memory (works even without torch)
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,
            "used": mem.used / 1024**3,
            "free": mem.available / 1024**3,
            "type": "cpu"
        }


def cleanup_device() -> None:
    """
    Clean up device memory/caches.
    """
    try:
        import torch
    except ImportError:
        return

    device_type = get_device_type()

    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    elif device_type == "mps":
        try:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    # Always do Python GC
    import gc
    gc.collect()


def is_gpu_available() -> bool:
    """
    Check if GPU (CUDA or MPS) is available.

    Returns:
        bool: True if GPU is available
    """
    device_type = get_device_type()
    return device_type in ("cuda", "mps")


def get_torch_device() -> str:
    """
    Get torch device string for model placement.

    Returns:
        str: Device string (e.g., "cuda", "mps", "cpu")
    """
    return get_device_type()


if __name__ == "__main__":
    # Test device detection
    print("=== ClarityOCR Device Detection ===")
    print(f"Device Type: {get_device_type()}")
    print(f"Device Name: {get_device_name()}")
    print(f"GPU Available: {is_gpu_available()}")

    config_msg = configure_device()
    if config_msg:
        print(f"\nConfiguration:")
        print(config_msg)

    mem = get_memory_info()
    print(f"\nMemory Info:")
    print(f"  Total: {mem['total']:.2f} GB")
    print(f"  Used:  {mem['used']:.2f} GB")
    print(f"  Free:  {mem['free']:.2f} GB")
