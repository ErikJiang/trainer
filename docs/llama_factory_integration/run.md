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

### Step 2 — 创建训练配置 ConfigMap

```bash
kubectl apply -f examples/llamafactory/lora-sft-configmap.yaml

# 确认 ConfigMap 已就绪
kubectl get configmap llamafactory-lora-sft-config -o yaml
```

说明：`dataset_info.json` 中建议使用 `"file_name": "/workspace/dataset/data"`（目录），不要写死 parquet 文件名。因为 HuggingFace 数据集导出的 parquet 名称可能包含哈希后缀。

### Step 3 — 提交 TrainJob

lora-sft-trainjob.yaml 中 `numNodes: 2` 会覆盖 ClusterTrainingRuntime 中的 `numNodes: 1`，对于 0.6B 的验证场景建议改为 1 节点：

```bash
# 单节点验证
kubectl apply -f - <<EOF
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: llamafactory-lora-sft-demo
spec:
  runtimeRef:
    name: llamafactory-lora-sft
  trainer:
    numNodes: 1
    resourcesPerNode:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
EOF
```

### Step 4 — 观察运行状态

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

### Step 5 — 验收

训练日志中出现如下输出即表示成功：
```
{'loss': ..., 'learning_rate': ..., 'epoch': ...}
...
Training completed. Do not forget to share your model on huggingface.co/models
```

### 清理

```bash
kubectl delete trainjob llamafactory-lora-sft-demo
kubectl delete configmap llamafactory-lora-sft-config
```