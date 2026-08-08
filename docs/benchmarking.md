# Benchmark methodology

Performance claims must use the same model weights, image size, preprocessing, device, and batch sizes.

## Procedure

1. Close unrelated GPU-heavy applications.
2. Load the model once.
3. Run at least five unmeasured warmup iterations.
4. Synchronize the accelerator around every timed iteration.
5. Run at least 30 measured iterations per batch size.
6. Report p50, p95, p99, mean latency, images/sec, process RSS, and accelerator memory.
7. Record software and hardware versions.
8. Repeat the benchmark at least three times and report variance for published results.

The included benchmark produces `artifacts/.../benchmark-<backend>.json`.

```bash
python scripts/benchmark.py --config configs/quickstart.yaml --batch-sizes 1,8,16,32
```

Change `inference.backend` to `onnx` after exporting the model, then run the same command. TensorRT results should follow the same warmup, shape, and reporting rules.

