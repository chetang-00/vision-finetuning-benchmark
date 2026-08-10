# Vision Fine-Tuning Benchmark

A reproducible deep-learning benchmark comparing CNN and Vision Transformer architectures across frozen, partial, and full fine-tuning strategies. It includes model evaluation, explainability, inference optimization, API deployment, and hardware performance measurements.

This repository is designed to answer a complete engineering question:

> Which vision architecture and fine-tuning strategy provides the best accuracy–latency–memory trade-off for a 101-class food classifier on Apple Silicon and NVIDIA GPUs?

No benchmark numbers are fabricated. The included commands generate the result artifacts from your own hardware.

## What is included

- Food-101 download, deterministic train/validation/test handling, subset quickstarts, and augmentation pipelines
- ResNet50, ResNet101, EfficientNet-B0, ConvNeXt-Tiny, ViT-Base, and ViT-Tiny through `timm`
- Frozen-backbone, partial, and full fine-tuning strategies
- AdamW, discriminative learning rates, label smoothing, MixUp, CutMix, gradient accumulation, gradient clipping, gradient checkpointing, warmup, cosine decay, and early stopping
- CUDA, Apple MPS, and CPU device selection
- Optional MLflow, TensorBoard, and Weights & Biases tracking
- Top-1/Top-5 accuracy, macro metrics, per-class evaluation, calibration error, and confusion matrix generation
- PyTorch and ONNX Runtime inference backends
- ONNX export and NVIDIA TensorRT engine build helper
- Latency, throughput, process memory, and accelerator-memory benchmarks
- FastAPI single-image and batch endpoints, health checks, service metrics, and Grad-CAM
- A responsive browser interface
- CPU and NVIDIA Docker profiles, tests, linting, and GitHub Actions

## System flow

```text
Food-101 → augmentation → model factory → fine-tuning → evaluation
                                      │
                                      ├── MLflow / TensorBoard / W&B
                                      └── checkpoint
                                             │
                    ┌────────────────────────┼────────────────────┐
                    ▼                        ▼                    ▼
             PyTorch inference        ONNX Runtime        TensorRT (CUDA)
                    └────────────────────────┼────────────────────┘
                                             ▼
                                  FastAPI + browser UI
                                             ▼
                                  reproducible benchmarks
```

More detail is available in [docs/architecture.md](docs/architecture.md).

## Quickstart on an M2 Pro

Python 3.11 is recommended. The system Python on older macOS installations may be too old.

```bash
cd vision-finetuning-benchmark
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[all]"
```

Confirm MPS availability:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Train the quickstart configuration. It uses 10% of Food-101, a frozen ResNet50 backbone, and three epochs so the full pipeline can be validated first:

```bash
python scripts/train.py --config configs/quickstart.yaml --device mps
```

Generate a dataset integrity and class-distribution report at any time:

```bash
python scripts/inspect_dataset.py --config configs/quickstart.yaml
```

Evaluate the best checkpoint:

```bash
python scripts/evaluate.py --config configs/quickstart.yaml --device mps
```

Start the application with the quickstart ResNet50 checkpoint:

```bash
FOODVISION_CONFIG=configs/quickstart.yaml \
FOODVISION_DEVICE=mps \
uvicorn foodvision.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. API documentation is available at `http://127.0.0.1:8000/docs`.

See [docs/macos-m2.md](docs/macos-m2.md) for memory and precision guidance.

## Full experiment plan

The repository includes a 4 × 3 model/fine-tuning matrix under `configs/generated/`. Regenerate those files after changing `configs/base.yaml` with:

```bash
python scripts/run_experiment_matrix.py --base configs/base.yaml
```

This creates or refreshes twelve configurations under model-specific folders in `configs/generated/`:

| Architecture | Frozen | Partial | Full |
|---|---:|---:|---:|
| ResNet50 | ✓ | ✓ | ✓ |
| EfficientNet-B0 | ✓ | ✓ | ✓ |
| ConvNeXt-Tiny | ✓ | ✓ | ✓ |
| ViT-Base/16 | ✓ | ✓ | ✓ |

Run one experiment at a time:

```bash
python scripts/train.py \
  --config configs/generated/convnext_tiny/partial.yaml \
  --device mps
```

Run the complete matrix only on hardware and within a compute budget you control:

```bash
python scripts/run_experiment_matrix.py --base configs/base.yaml --run
```

ResNet101 and ViT-Tiny remain available through the model factory for additional controlled experiments.

Every model and fine-tuning strategy has an isolated artifact directory:

```text
artifacts/
├── quickstart/resnet50-frozen/
├── resnet50/{frozen,partial,full}/
├── efficientnet_b0/{frozen,partial,full}/
├── convnext_tiny/{frozen,partial,full}/
└── vit_b_16/{frozen,partial,full}/
```

Training checkpoints, evaluation reports, benchmarks, and ONNX exports for one experiment stay together. A ResNet run cannot overwrite a ViT or EfficientNet run.

## Pretrained weights and trained checkpoints

With `model.pretrained: true`, `timm` downloads the original ImageNet weights once and stores them in the shared Hugging Face cache, normally under:

```text
~/.cache/huggingface/hub/models--timm--<model-name>/
```

That cache persists across terminal sessions and Mac restarts and can be reused by other projects running as the same user. It is not committed to this repository.

Training saves the complete experiment checkpoint separately:

```text
artifacts/<architecture>/<fine-tuning-strategy>/best.pt
```

`best.pt` contains the complete model state, Food-101 class list, resolved configuration, optimizer state, scheduler state, epoch, and best validation score. Even a frozen-backbone run saves the whole model so inference does not depend on separately copying the cached pretrained file.

## Configuration

Every run is defined by YAML. A checkpoint stores the resolved configuration and Food-101 class list with the weights.

Important settings:

```yaml
model:
  name: convnext_tiny
  fine_tune: partial       # frozen | partial | full
  unfreeze_blocks: 1
  gradient_checkpointing: false

training:
  precision: auto          # auto | fp16 | bf16 | fp32
  accumulation_steps: 2
  warmup_epochs: 2
  mixup_alpha: 0.2
  cutmix_alpha: 1.0
```

The automatic precision mode uses FP16 autocast on CUDA and conservative FP32 on MPS/CPU. MPS FP16 can be explicitly enabled after validating numerical stability for the chosen architecture.

## Tracking experiments

The default full configuration enables MLflow and TensorBoard. Start the MLflow UI after training:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

```bash
tensorboard --logdir artifacts
```

W&B is opt-in. Set `tracking.wandb: true`, authenticate through W&B, and keep credentials outside the repository.

Each run tracks configuration, train/validation loss, Top-1, Top-5, learning rate, checkpoints, and training history. Evaluation generates macro metrics, expected calibration error, per-class metrics, and a confusion matrix.

## API

One API process loads one model during startup. Select the model with `FOODVISION_CONFIG`; the referenced checkpoint must already exist. For example, serve the full-data ResNet50 frozen experiment with:

```bash
FOODVISION_CONFIG=configs/generated/resnet50/frozen.yaml \
FOODVISION_DEVICE=mps \
uvicorn foodvision.api.main:app --host 127.0.0.1 --port 8000
```

To switch to another architecture or fine-tuning strategy, stop Uvicorn with `Ctrl+C` and restart it with another configuration, such as `configs/generated/vit_b_16/full.yaml`. Use `GET /models` to confirm the active architecture, backend, and fine-tuning strategy.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness and active backend |
| `GET` | `/models` | Active model metadata |
| `GET` | `/metrics` | Lightweight service counters |
| `POST` | `/predict` | Single-image Top-K prediction |
| `POST` | `/predict/batch` | Batch prediction, up to 32 images |
| `POST` | `/explain` | Grad-CAM overlay for CNN models |

Example:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
  -F "image=@example.jpg"
```

The service validates MIME type, file integrity, file size, Top-K bounds, and batch size. It loads the model once during startup and returns a request ID, model/backend metadata, and measured inference time.

## ONNX Runtime

Export the selected PyTorch checkpoint:

```bash
python scripts/export_onnx.py --config configs/quickstart.yaml
```

Change the configuration:

```yaml
inference:
  backend: onnx
  onnx_model: artifacts/quickstart/resnet50-frozen/model.onnx
```

Then run the same API or benchmark command. Dynamic batch axes are included in the ONNX graph.

## Benchmarking

Benchmark batch sizes 1, 8, 16, and 32:

```bash
python scripts/benchmark.py \
  --config configs/quickstart.yaml \
  --batch-sizes 1,8,16,32 \
  --iterations 50
```

The report includes mean, p50, p95, p99, images/sec, process RSS, and accelerator memory. Accelerator synchronization prevents misleading asynchronous timings.

The benchmark uses generated input tensors and measures raw model execution. It does not include JPEG decoding, image preprocessing, HTTP transfer, or browser time.

Use the same model weights, precision, image size, and hardware when comparing backends. Full methodology is in [docs/benchmarking.md](docs/benchmarking.md).

## CUDA and TensorRT

The CUDA path is intentionally separate from the Apple-Silicon path:

```bash
docker compose --profile cuda up --build
```

After ONNX export, use `scripts/build_tensorrt.py` in an NVIDIA TensorRT environment:

```bash
python scripts/build_tensorrt.py \
  --onnx artifacts/resnet50/partial/model.onnx \
  --output artifacts/resnet50/partial/model-fp16.engine \
  --precision fp16
```

TensorRT cannot run on an M2 Pro. Build and benchmark its engine on the target NVIDIA system. See [docs/cuda.md](docs/cuda.md).

## Docker CPU service

After a checkpoint exists:

```bash
docker compose --profile cpu up --build
```

The local `artifacts/` directory is mounted read-only into the service.

## Testing and quality

```bash
pytest --cov=foodvision --cov-report=term-missing
ruff check .
mypy src/foodvision
```

CI runs linting, tests, coverage, and a CPU-container build. Tests intentionally do not download Food-101 or pretrained weights.

## Repository layout

```text
├── configs/                    # quickstart, full, and model configurations
├── docker/                     # CPU and CUDA images
├── docs/                       # architecture, platform guides, model card
├── scripts/                    # operational entry points
├── src/foodvision/
│   ├── api/                    # FastAPI service
│   ├── benchmarking.py          # repeatable inference measurements
│   ├── data/                   # Food-101 and transforms
│   ├── evaluation/             # test reports and calibration
│   ├── explainability/         # Grad-CAM
│   ├── export/                 # ONNX
│   ├── inference/              # backend-independent predictor
│   ├── models/                 # model and fine-tuning factory
│   ├── training/               # trainer, schedule, metrics, tracking
│   └── web/                    # browser interface
└── tests/
```

## Responsible use

The classifier is not an ingredient detector and must not be used for allergy, medical, food-safety, or religious-diet decisions. It is a closed-set model and will force unknown inputs into one of 101 known classes. See [docs/model-card.md](docs/model-card.md).

## Measured quickstart baseline

The following values were generated by the included scripts on an Apple M2 Pro using PyTorch MPS. This is a pipeline-validation baseline trained for three epochs on 10% of Food-101 with a frozen ResNet50 backbone; it is not the final full-dataset comparison.

### Model quality

| Metric | Result |
|---|---:|
| Test samples | 2,525 |
| Test Top-1 accuracy | 32.24% |
| Test Top-5 accuracy | 60.91% |
| Macro F1 | 29.34% |
| Expected calibration error | 0.2789 |

### Raw PyTorch MPS inference

| Batch | Mean latency | p95 latency | Throughput | Accelerator memory |
|---:|---:|---:|---:|---:|
| 1 | 6.67 ms | 6.90 ms | 149.9 images/s | 91.5 MB |
| 8 | 30.89 ms | 31.40 ms | 259.0 images/s | 98.7 MB |
| 16 | 60.64 ms | 61.11 ms | 263.9 images/s | 102.9 MB |

Published model comparisons should use the full dataset, a fixed seed and split, the same hardware and precision, and identical benchmark settings across architectures.

## Suggested published results

After running the experiments, update the repository with:

- Model × fine-tuning accuracy table
- Accuracy versus latency chart
- Throughput versus batch-size chart
- Memory versus batch-size chart
- Confusion-matrix image
- Five representative failure cases
- Hardware/software manifest
- Final model-card measurements

These artifacts turn the repository from source code into an evidence-backed engineering case study.
