# Running on Apple Silicon

## Recommended environment

Use Python 3.11 in a native arm64 terminal. Create an isolated environment and install the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

Confirm that PyTorch sees Metal:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Run the quickstart using 10% of Food-101 and a frozen ResNet50 backbone:

```bash
python scripts/train.py --config configs/quickstart.yaml --device mps
```

Full Food-101 experiments are configured in `configs/base.yaml`. Start with batch size 16. If memory pressure occurs, reduce the batch size to 8 and increase `accumulation_steps` to preserve the effective batch size.

## Apple-specific limitations

- CUDA and TensorRT do not run on Apple Silicon.
- Automatic mixed precision is kept conservative by default on MPS. Set `training.precision: fp16` only after confirming the selected model is stable.
- `torch.compile` is intentionally disabled in the predictor on MPS because support varies by operation and PyTorch release.
- MPS reports allocated memory, but it does not expose the same utilization telemetry as `nvidia-smi`.

