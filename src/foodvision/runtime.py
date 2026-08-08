from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    name: str
    accelerator: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preference: str = "auto") -> RuntimeInfo:
    preference = preference.lower()
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return RuntimeInfo(torch.device("cuda"), torch.cuda.get_device_name(0), "cuda")
    if preference in {"auto", "mps"} and torch.backends.mps.is_available():
        return RuntimeInfo(torch.device("mps"), "Apple Metal Performance Shaders", "mps")
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device preference: {preference}")
    if preference in {"cuda", "mps"}:
        raise RuntimeError(f"Requested device '{preference}' is unavailable")
    return RuntimeInfo(torch.device("cpu"), "CPU", "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0.0
