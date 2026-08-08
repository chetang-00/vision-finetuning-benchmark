# Architecture

FoodVision separates training-time and serving-time concerns while keeping one model contract.

```text
Food-101
   │
   ├── deterministic split + evaluation transforms
   └── augmented training pipeline
             │
             ▼
  timm model factory ── frozen / partial / full fine-tuning
             │
             ▼
  PyTorch trainer ── AMP ── gradient accumulation ── warmup + cosine
             │
       ┌─────┼──────────────┐
       ▼     ▼              ▼
 checkpoints  MLflow      TensorBoard / W&B
       │
       ├── PyTorch eager / compiled inference
       ├── ONNX Runtime
       └── TensorRT engine on NVIDIA
               │
               ▼
       FastAPI + browser UI
```

## Design decisions

- **Configuration-driven:** every experiment is represented by a YAML file and stored with its checkpoint.
- **One model factory:** training, evaluation, export, and inference reconstruct architectures through the same code path.
- **Backend boundary:** `Predictor` hides PyTorch versus ONNX execution from the API.
- **Portable accelerator selection:** CUDA is preferred, then MPS, then CPU.
- **Lazy optional integrations:** MLflow, W&B, TensorBoard, and ONNX packages are required only when enabled.
- **Measured claims:** benchmark output is emitted as JSON; the README contains no invented performance numbers.

## Artifact contract

Each checkpoint contains model weights, optimizer state, scheduler state, resolved configuration, class names, architecture, epoch, and validation score. This makes inference reproducible without hidden state.

Artifacts are isolated by architecture and fine-tuning strategy:

```text
artifacts/<model>/<strategy>/
├── best.pt
├── last.pt
├── history.json
├── resolved_config.json
├── evaluation.json
├── benchmark-pytorch.json
└── model.onnx
```

The quickstart is kept separately under `artifacts/quickstart/resnet50-frozen/`. The matrix generator updates the training output, inference checkpoint, and ONNX destination together so paths cannot drift between stages.
