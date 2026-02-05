from __future__ import annotations

import os
from typing import Optional

_DLL_DIR_HANDLES = []


def _gpu_id() -> int:
    try:
        return int(os.getenv("GPU_ID", "0"))
    except ValueError:
        return 0


def _ensure_cuda_bin_in_path() -> None:
    candidates = []
    local_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cuda_bin"))
    candidates.append(local_bin)
    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))
    cuda_path_118 = os.getenv("CUDA_PATH_V11_8")
    if cuda_path_118:
        candidates.append(os.path.join(cuda_path_118, "bin"))
    candidates.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")

    current = os.environ.get("PATH", "")
    parts = [p.lower() for p in current.split(";") if p]
    for cand in candidates:
        if not cand or not os.path.isdir(cand):
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(cand))
            except OSError:
                pass
        if cand.lower() not in parts:
            os.environ["PATH"] = f"{cand};{current}"
            parts.append(cand.lower())


def _torch_check(device_id: int) -> float:
    try:
        import torch
    except Exception as exc:  
        raise RuntimeError("GPU-only mode requires torch with CUDA support installed.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU-only mode: torch CUDA is not available. Install a CUDA-enabled torch build."
        )
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    device_count = torch.cuda.device_count()
    if device_count <= device_id:
        raise RuntimeError(f"GPU-only mode: CUDA device {device_id} not found (count={device_count}).")

    try:
        torch.cuda.set_device(device_id)
    except Exception:
        pass

    props = torch.cuda.get_device_properties(device_id)
    return props.total_memory / (1024 ** 3)


def _paddle_check(device_id: int) -> None:
    try:
        import paddle
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("GPU-only mode requires paddlepaddle-gpu installed.") from exc

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(
            "GPU-only mode: PaddlePaddle is not compiled with CUDA. Install paddlepaddle-gpu."
        )

    try:
        count = paddle.device.cuda.device_count()
    except Exception:
        count = 0
    if count <= device_id:
        raise RuntimeError(f"GPU-only mode: Paddle CUDA device {device_id} not found (count={count}).")


def ensure_gpu_available() -> float:
    _ensure_cuda_bin_in_path()
    device_id = _gpu_id()
    vram_gb = _torch_check(device_id)
    _paddle_check(device_id)
    return vram_gb


def require_gpu() -> int:
    if os.getenv("CV_USE_GPU", "1") != "1":
        raise RuntimeError("GPU-only mode: CV_USE_GPU must be 1.")
    ensure_gpu_available()
    return _gpu_id()


def vram_limit_mb(default: int = 2000) -> int:
    try:
        return int(os.getenv("CV_GPU_MEM", str(default)))
    except ValueError:
        return default
