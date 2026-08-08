from __future__ import annotations

import json
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import psutil
import torch

from foodvision.config import AppConfig
from foodvision.inference import Predictor
from foodvision.runtime import peak_memory_mb, synchronize


def run_benchmark(
    config: AppConfig,
    batch_sizes: list[int],
    iterations: int = 30,
    compile_model: bool = False,
) -> dict[str, Any]:
    predictor = Predictor(config, compile_model=compile_model)
    results = []
    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, 3, config.data.image_size, config.data.image_size)
        if predictor.backend == "pytorch":
            inputs = inputs.to(predictor.runtime.device)
            call = partial(predictor.model, inputs)
        else:
            call = partial(
                predictor.session.run,
                None,
                {predictor.input_name: inputs.numpy()},
            )

        with torch.inference_mode():
            for _ in range(config.inference.warmup_iterations):
                call()
            synchronize(predictor.runtime.device)
            latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                call()
                synchronize(predictor.runtime.device)
                latencies.append((time.perf_counter() - start) * 1000)

        ordered = sorted(latencies)
        mean_ms = statistics.mean(latencies)
        results.append(
            {
                "batch_size": batch_size,
                "mean_ms": mean_ms,
                "p50_ms": _percentile(ordered, 0.50),
                "p95_ms": _percentile(ordered, 0.95),
                "p99_ms": _percentile(ordered, 0.99),
                "images_per_second": batch_size / (mean_ms / 1000),
                "accelerator_memory_mb": peak_memory_mb(predictor.runtime.device),
                "process_rss_mb": psutil.Process().memory_info().rss / (1024**2),
            }
        )
    report = {
        "backend": predictor.backend,
        "device": predictor.runtime.name,
        "model": config.model.name,
        "iterations": iterations,
        "results": results,
    }
    output = Path(config.project.output_dir) / f"benchmark-{predictor.backend}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return values[min(len(values) - 1, int(round((len(values) - 1) * fraction)))]
