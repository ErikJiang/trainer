## LlamaFactory LoRA SFT 验证步骤

**前提**：集群中已安装 Trainer controller 及 CRD。

### Step 1 — 部署 ClusterTrainingRuntime

```bash
# 通过 overlay 部署（tag 会被正确替换）
kubectl apply -k manifests/overlays/runtimes/

# 确认 llamafactory-lora-sft 已创建
kubectl get clustertrainingruntime llamafactory-lora-sft

# 确认 pvc 已创建
kubectl get pvc llamafactory-lora-sft
```

### Step 2 — 创建 Demo Manifest（ConfigMap + TrainJob）

```bash
kubectl apply -f examples/llamafactory/lora-sft-demo.yaml

# 确认 ConfigMap 与 TrainJob 已创建
kubectl get configmap llamafactory-lora-sft-config -o yaml
kubectl get trainjob llamafactory-lora-sft-demo
```

说明：`dataset_info.json` 中建议使用 `"file_name": "/workspace/dataset/data"`（目录），不要写死 parquet 文件名。因为 HuggingFace 数据集导出的 parquet 名称可能包含哈希后缀。

当前示例默认使用单节点、单 GPU 验证；如需调整训练参数或节点资源，直接修改 [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中对应的 ConfigMap 或 TrainJob 段。

### Step 3 — 观察运行状态

```bash
# 查看 TrainJob 整体状态
kubectl get trainjob llamafactory-lora-sft-demo -w

# 查看 JobSet 和各阶段 Pod
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=llamafactory-lora-sft-demo

# 各阶段日志（按顺序执行）
# 1. 数据集下载
kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=dataset-initializer --tail=50

# 2. 模型下载（Qwen3-0.6B ~1.3GB）
kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=model-initializer --tail=50

# 3. 训练进度（关注 loss 下降）
kubectl logs -l trainer.kubeflow.org/trainjob-ancestor-step=trainer -f
```

### Step 4 — 验收

训练日志中出现如下输出即表示成功：
```
{'loss': ..., 'learning_rate': ..., 'epoch': ...}
...
Training completed. Do not forget to share your model on huggingface.co/models
```

### 清理

```bash
kubectl delete -f examples/llamafactory/lora-sft-demo.yaml
```