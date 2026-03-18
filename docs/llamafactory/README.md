# LlamaFactory Integration on Kubeflow Trainer

## Overview

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) is integrated into Kubeflow Trainer
as a builtin trainer through the Torch Plugin, providing native distributed training support
with automatic environment variable bridging and validation.

**Key Features:**

- Automatic bridging of Kubeflow `PET_*` environment variables to LlamaFactory's expected format
- Reserved environment variable validation to prevent user conflicts
- `volumeClaimPolicies` for automatic PVC lifecycle management
- Pre-built `ClusterTrainingRuntime` for LoRA SFT with Qwen2.5-1.5B-Instruct

## Architecture

```
User submits TrainJob
    │
    ▼
Kubeflow Trainer Controller
    │
    ├─ Reads ClusterTrainingRuntime "llamafactory-lora-sft"
    ├─ Runs Torch Plugin (EnforceMLPolicy)
    │   ├─ Injects PET_MASTER_ADDR, PET_MASTER_PORT
    │   └─ Bridges PET_* → NNODES, MASTER_ADDR, etc. + FORCE_TORCHRUN=1
    ├─ Creates JobSet
    │   ├─ Job: dataset-initializer  → Downloads data to PVC /workspace/dataset
    │   ├─ Job: model-initializer    → Downloads model to PVC /workspace/model
    │   └─ Job: node (replicas=numNodes)
    │       ├─ Pod 0 (NODE_RANK=0): llamafactory-cli train /config/training.yaml
    │       │   └─ torchrun → training process (master)
    │       └─ Pod 1 (NODE_RANK=1): llamafactory-cli train /config/training.yaml
    │           └─ torchrun → training process (worker)
    │
    ▼
Training complete → checkpoint saved to PVC /workspace/output
```

## How It Works

### Environment Variable Bridging

Kubeflow's Torch Plugin injects `PET_*` prefixed environment variables (PyTorch Elastic convention).
LlamaFactory's launcher reads unprefixed variables (`NNODES`, `MASTER_ADDR`, etc.).

The LlamaFactory plugin in `torch.go` automatically bridges these:

| Kubeflow Injected | LlamaFactory Expected | Source |
|---|---|---|
| `PET_NNODES` | `NNODES` | `trainJob.spec.trainer.numNodes` |
| `PET_NPROC_PER_NODE` | `NPROC_PER_NODE` | GPU count or `"auto"` |
| `PET_NODE_RANK` | `NODE_RANK` | `batch.kubernetes.io/job-completion-index` |
| `PET_MASTER_ADDR` | `MASTER_ADDR` | `{trainjob}-node-0-0.{trainjob}` |
| `PET_MASTER_PORT` | `MASTER_PORT` | `29400` |

Additionally, `FORCE_TORCHRUN=1` is injected to ensure LlamaFactory uses `torchrun` for
distributed training (even for single-GPU setups within each node).

### Configuration via ConfigMap

LlamaFactory uses YAML configuration files and a `dataset_info.json` file.
These are provided via a Kubernetes ConfigMap mounted at `/config/`:

- `/config/training.yaml` — Training hyperparameters (model path, LoRA config, dataset, etc.)
- `/config/dataset_info.json` — Dataset metadata pointing to the downloaded data path

This approach aligns with LlamaFactory's native configuration style, unlike CLI-arg-based
trainers (e.g., TorchTune) where parameters can be passed directly on the command line.

### Entrypoint Detection

The Torch Plugin detects LlamaFactory by checking if the container command starts with
`["llamafactory-cli", "train"]`. When detected:

1. `PET_MASTER_ADDR` and `PET_MASTER_PORT` are injected (same as standard torchrun path)
2. All 5 bridge variables + `FORCE_TORCHRUN=1` are injected
3. Reserved environment variable names (`NNODES`, `NPROC_PER_NODE`, `NODE_RANK`,
   `MASTER_ADDR`, `MASTER_PORT`, `FORCE_TORCHRUN`) are protected from user override

## Quick Start

### 1. Deploy the ClusterTrainingRuntime

The `llamafactory-lora-sft` runtime is included in the default Kubeflow Trainer installation.
Verify it exists:

```bash
kubectl get clustertrainingruntime llamafactory-lora-sft
```

### 2. Create the Training Configuration

```bash
kubectl apply -f examples/llamafactory/lora-sft-configmap.yaml
```

This creates a ConfigMap with:
- LoRA SFT configuration for Qwen2.5-1.5B-Instruct
- Alpaca dataset metadata

### 3. Submit a TrainJob

```bash
kubectl apply -f examples/llamafactory/lora-sft-trainjob.yaml
```

This creates a 2-node distributed LoRA SFT training job with 1 GPU per node.

### 4. Monitor Training

```bash
# Watch TrainJob status
kubectl get trainjob llamafactory-lora-sft-demo -w

# View training logs
kubectl logs -l batch.kubernetes.io/job-name=llamafactory-lora-sft-demo-node-0 -f
```

## Customization

### Change Model

Update the `model-initializer` STORAGE_URI in a custom runtime or override via TrainJob:

```yaml
spec:
  modelConfig:
    input:
      storageUri: hf://meta-llama/Llama-3.2-1B-Instruct
```

Then update `model_name_or_path` in the ConfigMap accordingly.

### Change Dataset

Update the `dataset-initializer` STORAGE_URI and adjust `dataset_info.json` in the ConfigMap
to point to the correct data file path.

### Adjust Training Parameters

Modify the ConfigMap's `training.yaml` to change hyperparameters like:
- `lora_rank`, `lora_target` — LoRA configuration
- `per_device_train_batch_size`, `gradient_accumulation_steps` — Batch settings
- `learning_rate`, `num_train_epochs` — Training schedule
- `bf16` — Mixed precision

### Scale Nodes

Change `numNodes` in the TrainJob spec:

```yaml
spec:
  trainer:
    numNodes: 4
    resourcesPerNode:
      limits:
        nvidia.com/gpu: 2
```

## Files Reference

| File | Description |
|------|-------------|
| `cmd/trainers/llamafactory/Dockerfile` | Minimal Dockerfile based on `hiyouga/llamafactory:0.9.4` |
| `pkg/runtime/framework/plugins/torch/llamafactory.go` | Plugin logic for env bridging |
| `pkg/constants/constants.go` | LlamaFactory constants (env names, entrypoint, reserved set) |
| `manifests/base/runtimes/llamafactory/` | ClusterTrainingRuntime manifests |
| `examples/llamafactory/` | Example ConfigMap and TrainJob |

## Future Work

- **SDK Integration**: `LlamaFactoryConfig` for the Kubeflow Python SDK
- **Additional Runtimes**: Full SFT, DPO/RLHF, DeepSpeed ZeRO
- **Multi-modal Training**: Vision-language model support
