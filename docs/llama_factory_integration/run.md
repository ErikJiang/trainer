## LlamaFactory LoRA SFT 验证步骤

**前提**：集群中可运行普通 Kubernetes Job，且节点具备 1 张可用 GPU。

### Step 1 — 创建 Demo Manifest（PVC + ConfigMap + Job）

```bash
kubectl apply -f examples/llamafactory/lora-sft-demo.yaml

# 确认 PVC、ConfigMap 与 Job 已创建
kubectl get pvc llamafactory-lora-sft-demo
kubectl get configmap llamafactory-lora-sft-config -o yaml
kubectl get job llamafactory-lora-sft-demo
```

说明：`dataset_info.json` 中建议使用 `"file_name": "/workspace/dataset/data"`（目录），不要写死 parquet 文件名。因为 HuggingFace 数据集导出的 parquet 名称可能包含哈希后缀。

当前示例默认使用单节点、单 GPU 验证；如需调整训练参数或节点资源，直接修改 [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中对应的 ConfigMap 或 Job 段。

当前 Job 直接调用 `torchrun --standalone --nnodes=1 --nproc_per_node=1`，不依赖 Trainer 的 Torch Plugin，因此不需要 `PET_*` 到 `NNODES`/`MASTER_ADDR` 的桥接。

### Step 2 — 观察运行状态

```bash
# 查看 Job 整体状态
kubectl get job llamafactory-lora-sft-demo -w

# 查看 Pod
kubectl get pods -l app.kubernetes.io/name=llamafactory-lora-sft-demo

# 查看 initContainer 日志
kubectl logs job/llamafactory-lora-sft-demo -c dataset-initializer
kubectl logs job/llamafactory-lora-sft-demo -c model-initializer

# 查看训练进度（关注 loss 下降）
kubectl logs job/llamafactory-lora-sft-demo -c trainer -f
```

### Step 3 — 验收

训练日志中出现如下输出即表示成功：
```
{'loss': ..., 'learning_rate': ..., 'epoch': ...}
...
Training completed. Do not forget to share your model on huggingface.co/models
```

如果需要确认确实走的是 `torchrun`，可以检查日志开头是否包含当前示例打印的以下标记：

```
[job] Launching single-node LoRA SFT with torchrun
```

### 清理

```bash
kubectl delete -f examples/llamafactory/lora-sft-demo.yaml
```