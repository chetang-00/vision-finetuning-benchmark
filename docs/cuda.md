# NVIDIA CUDA and TensorRT

Run training in the CUDA service or a native NVIDIA environment:

```bash
docker compose --profile cuda run --rm foodvision-cuda \
  python -m foodvision.training.cli --config configs/base.yaml --device cuda
```

Export the selected checkpoint:

```bash
python scripts/export_onnx.py --config configs/base.yaml --checkpoint artifacts/best.pt
```

Build a TensorRT engine inside an NVIDIA TensorRT container that provides `trtexec`:

```bash
python scripts/build_tensorrt.py \
  --onnx artifacts/model.onnx \
  --output artifacts/model-fp16.engine \
  --precision fp16 \
  --max-batch 32
```

TensorRT engines are target-specific artifacts. Record the GPU, driver, CUDA, TensorRT, precision, and shape profile alongside every benchmark result. Do not commit engine binaries to Git.

