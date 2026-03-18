# LlamaFactory Examples

This directory contains examples for running LlamaFactory LoRA SFT training on Kubeflow Trainer.

## Prerequisites

- A Kubernetes cluster with Kubeflow Trainer installed
- GPU nodes available in the cluster
- The `llamafactory-lora-sft` ClusterTrainingRuntime deployed

## Files

- **lora-sft-configmap.yaml**: ConfigMap containing the LlamaFactory training configuration
  (training.yaml) and dataset metadata (dataset_info.json) for LoRA SFT with Qwen2.5-1.5B-Instruct
  on the Alpaca dataset.
- **lora-sft-trainjob.yaml**: TrainJob that references the `llamafactory-lora-sft` runtime
  to run distributed LoRA SFT training across 2 nodes.

## Usage

1. Create the training configuration:

```bash
kubectl apply -f lora-sft-configmap.yaml
```

2. Submit the TrainJob:

```bash
kubectl apply -f lora-sft-trainjob.yaml
```

3. Monitor training progress:

```bash
kubectl get trainjob llamafactory-lora-sft-demo -w
kubectl logs -l batch.kubernetes.io/job-name=llamafactory-lora-sft-demo-node-0 -f
```
