# LLaMA Factory on Kubeflow Trainer 集成设计方案

## 概述

本文档描述如何将 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 微调框架集成到 Kubeflow Trainer 平台上运行，分为两个阶段：

- **Phase 1**：使用 CustomTrainer + ClusterTrainingRuntime（纯 YAML/Docker，不修改 Go 代码），快速验证可行性
- **Phase 2**：作为 BuiltinTrainer 深度集成（修改 Go 代码 + SDK），提供一等公民体验

**核心可行性依据**：LLaMA Factory 使用标准 `torchrun` 进行分布式训练，与 Kubeflow Trainer 的 Torch Plugin 协议层完全对齐。Kubeflow Trainer ROADMAP [issue #2752](https://github.com/kubeflow/trainer/issues/2752) 已将 LLaMA Factory 列为待探索的微调库。

**参考实现**：TorchTune 已完整集成到 Kubeflow Trainer 中（Dockerfile、ClusterTrainingRuntime、Torch Plugin、SDK），是本方案的核心参考模板。

---

## 关键设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| LLaMA Factory 版本 | **Legacy**（非 V1） | 更稳定、100+ 模型支持、文档更丰富，分布式协议与 V1 一致 |
| 训练阶段 | **LoRA SFT** | 最主流微调方式，显存需求低，适合小资源验证 |
| 分布式模式 | **DDP**（Phase 1） | 2 节点 × 1 GPU + LoRA 小模型，无需 ZeRO/FSDP 模型分片 |
| Runtime 类型 | **Torch Runtime** | 非 DeepSpeed/MPI Runtime；如需 DeepSpeed ZeRO 由 LLaMA Factory 内部管理 |
| 配置传递 | **ConfigMap → YAML 文件** | 最符合 LLaMA Factory 使用习惯 |

---

## 启动方式的关键技术问题与解决方案

### 问题：PET_ 环境变量名称不匹配

Kubeflow Torch Plugin 注入的环境变量使用 `PET_` 前缀（PyTorch Elastic 约定）：

| Kubeflow 注入的变量 | 值来源 |
|---------------------|--------|
| `PET_NNODES` | `trainJob.spec.trainer.numNodes` |
| `PET_NPROC_PER_NODE` | GPU 数或 `"auto"` |
| `PET_NODE_RANK` | `metadata.annotations[batch.kubernetes.io/job-completion-index]` |
| `PET_MASTER_ADDR` | `{trainjob-name}-node-0-0.{trainjob-name}` (JobSet headless service DNS) |
| `PET_MASTER_PORT` | `29400` |

> 参见 `pkg/constants/constants.go` 中 `TorchEnvNumNodes = "PET_NNODES"` 等定义。

而 LLaMA Factory Legacy launcher 读取的是**无前缀**的环境变量：

| LLaMA Factory 读取的变量 | 默认值 |
|--------------------------|--------|
| `NNODES` | `"1"` |
| `NODE_RANK` | `"0"` |
| `NPROC_PER_NODE` | `get_device_count()` |
| `MASTER_ADDR` | `"127.0.0.1"` |
| `MASTER_PORT` | 自动查找可用端口 |

> 参见 `LlamaFactory-repo/src/llamafactory/launcher.py` 第 64-77 行。

TorchTune 没有这个问题，因为 `tune run` 内部直接使用 PyTorch Elastic API 识别 `PET_*` 变量。

### 解决方案：Shell 包装脚本桥接环境变量

在容器 command 中使用 shell 脚本，将 `PET_*` 环境变量映射到 LLaMA Factory 期望的变量名：

```bash
#!/bin/sh
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export NPROC_PER_NODE=${PET_NPROC_PER_NODE:-1}
export MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${PET_MASTER_PORT:-29400}
export FORCE_TORCHRUN=1

llamafactory-cli train /config/training.yaml
```

### LLaMA Factory 的 torchrun 启动流程

理解 launcher 行为对避免"双层 torchrun"冲突至关重要：

```
llamafactory-cli train config.yaml
  → cli.py:main() → launcher.launch()
  → 弹出 "train" 命令，检查分布式条件：
    if FORCE_TORCHRUN=1 OR (GPU数 > 1 且不用 Ray):
      → 调用 torchrun 重新执行 launcher.py [剩余参数]
         ↓  (torchrun 为每个 GPU 启动一个进程)
         launcher.py 作为 __main__ 运行
           → run_exp() → run_sft() → HuggingFace Trainer.train()
    else:
      → 直接调用 run_exp()
```

> 参见 launcher.py 第 60-130 行（torchrun 调用）和第 182-185 行（`__main__` 入口）。

**在 Kubeflow 环境中**：设置 `FORCE_TORCHRUN=1` 后，LLaMA Factory 会读取桥接后的 `NNODES`、`MASTER_ADDR` 等变量来配置 torchrun，然后 torchrun 在每个 Pod 内启动 `NPROC_PER_NODE` 个训练进程。Kubeflow 的 JobSet 负责多节点编排（Pod 调度、DNS 服务发现），LLaMA Factory 的 torchrun 负责节点内多进程管理。

---

## Phase 1：CustomTrainer 快速集成

**目标**：不修改任何 Go 代码，纯通过 Docker 镜像 + YAML 配置，在 Kubeflow Trainer 上运行 LLaMA Factory LoRA SFT 分布式训练。

**验证环境**：2 节点 × 1 GPU，小模型（Qwen2.5-1.5B 或 Llama-3.2-1B）

### Step 1：构建 LLaMA Factory 训练镜像

**文件**：`cmd/trainers/llamafactory/Dockerfile`

基于 LLaMA Factory 官方 CUDA 镜像构建（参考 Dockerfile），添加启动包装脚本：

```dockerfile
FROM docker.io/hiyouga/llamafactory:latest

WORKDIR /workspace

# 添加 PET_ 环境变量桥接脚本
COPY cmd/trainers/llamafactory/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

**文件**：`cmd/trainers/llamafactory/entrypoint.sh`

```bash
#!/bin/sh
set -e

# 桥接 Kubeflow PET_ 环境变量 → LLaMA Factory 期望的变量名
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export NPROC_PER_NODE=${PET_NPROC_PER_NODE:-1}
export MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${PET_MASTER_PORT:-29400}
export FORCE_TORCHRUN=1

echo "[entrypoint] Distributed config: NNODES=$NNODES, NODE_RANK=$NODE_RANK, NPROC_PER_NODE=$NPROC_PER_NODE, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# 执行传入的命令（默认为 llamafactory-cli train）
exec "$@"
```

**构建命令**：
```bash
docker build . -f cmd/trainers/llamafactory/Dockerfile -t llamafactory-trainer:latest
```

> **参考**：TorchTune Dockerfile 位于 Dockerfile，结构类似但更简单（无需桥接脚本）。

### Step 2：创建 ClusterTrainingRuntime

**文件**：`manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml`

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: llamafactory-lora-sft
  labels:
    trainer.kubeflow.org/framework: llamafactory
spec:
  mlPolicy:
    numNodes: 1
    torch: {}
  template:
    spec:
      volumeClaimPolicies:
        - templates:
            - metadata:
                name: initializer
              spec:
                accessModes: ["ReadWriteOnce"]
                resources:
                  requests:
                    storage: 50Gi
      replicatedJobs:
        # --- 数据集初始化 ---
        - name: dataset-initializer
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: dataset-initializer
            spec:
              template:
                spec:
                  containers:
                    - name: dataset-initializer
                      image: ghcr.io/kubeflow/trainer/dataset-initializer
                      env:
                        - name: STORAGE_URI
                          value: hf://tatsu-lab/alpaca
                      volumeMounts:
                        - mountPath: /workspace
                          name: initializer
        # --- 模型初始化 ---
        - name: model-initializer
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: model-initializer
            spec:
              template:
                spec:
                  containers:
                    - name: model-initializer
                      image: ghcr.io/kubeflow/trainer/model-initializer
                      env:
                        - name: STORAGE_URI
                          value: hf://Qwen/Qwen2.5-1.5B
                      volumeMounts:
                        - mountPath: /workspace
                          name: initializer
        # --- 训练节点 ---
        - name: node
          dependsOn:
            - name: dataset-initializer
              status: Complete
            - name: model-initializer
              status: Complete
          template:
            metadata:
              labels:
                trainer.kubeflow.org/trainjob-ancestor-step: trainer
            spec:
              template:
                spec:
                  containers:
                    - name: node
                      image: llamafactory-trainer:latest
                      command:
                        - llamafactory-cli
                        - train
                        - /config/training.yaml
                      resources:
                        limits:
                          nvidia.com/gpu: 1
                      volumeMounts:
                        - mountPath: /workspace
                          name: initializer
                        - mountPath: /config
                          name: training-config
                  volumes:
                    - name: training-config
                      configMap:
                        name: llamafactory-lora-sft-config
```

> **参考模板**：llama3_2_1B.yaml，结构一致（dataset-initializer → model-initializer → node），区别在于 node 容器命令和额外的 ConfigMap volume。

### Step 3：创建训练配置 ConfigMap

**文件**：`examples/llamafactory/lora-sft-configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llamafactory-lora-sft-config
data:
  training.yaml: |
    ### model
    model_name_or_path: /workspace/model
    trust_remote_code: true

    ### method
    stage: sft
    do_train: true
    finetuning_type: lora
    lora_rank: 8
    lora_target: all

    ### dataset
    dataset_dir: /workspace/dataset
    dataset: alpaca_en_demo
    template: default
    cutoff_len: 1024
    max_samples: 500
    preprocessing_num_workers: 4

    ### output
    output_dir: /workspace/output
    logging_steps: 10
    save_steps: 500
    overwrite_output_dir: true

    ### train
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    learning_rate: 1.0e-4
    num_train_epochs: 1.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000

  dataset_info.json: |
    {
      "alpaca_en_demo": {
        "file_name": "/workspace/dataset/data/train-00000-of-00001.parquet"
      }
    }
```

> **关键配置说明**：
> - `model_name_or_path: /workspace/model`：指向 Model Initializer 下载的本地路径（对应 `constants.ModelMountPath = "/workspace/model"`）
> - `dataset_dir: /workspace/dataset`：指向 Dataset Initializer 下载的本地路径（对应 `constants.DatasetMountPath = "/workspace/dataset"`）
> - `finetuning_type: lora`：LoRA 微调，显存需求低
> - `ddp_timeout: 180000000`：分布式训练超时设置，避免节点启动时延导致超时

### Step 4：提交 TrainJob

**文件**：`examples/llamafactory/lora-sft-trainjob.yaml`

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: llamafactory-lora-sft-demo
spec:
  runtimeRef:
    name: llamafactory-lora-sft
  trainer:
    numNodes: 2
    resourcesPerNode:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
```

**提交命令**：
```bash
# 先创建 ConfigMap
kubectl apply -f examples/llamafactory/lora-sft-configmap.yaml

# 创建 TrainJob
kubectl apply -f examples/llamafactory/lora-sft-trainjob.yaml
```

### Step 5：验证

**验证清单**：

1. **Pod 启动顺序**：dataset-initializer 和 model-initializer 先完成 → node pods 启动
   ```bash
   kubectl get pods -l batch.kubernetes.io/job-name -w
   ```

2. **分布式初始化**：检查 node pod 日志中的分布式环境变量和 torchrun 初始化
   ```bash
   kubectl logs {trainjob-name}-node-0-0 | grep -E "Distributed|NNODES|MASTER_ADDR|rank"
   ```
   期望输出：
   ```
   [entrypoint] Distributed config: NNODES=2, NODE_RANK=0, NPROC_PER_NODE=1, MASTER_ADDR=..., MASTER_PORT=29400
   Initializing 1 distributed tasks at: {master-addr}:29400
   Multi-node training enabled: num nodes: 2, node rank: 0
   ```

3. **训练进行中**：确认 loss 在下降
   ```bash
   kubectl logs {trainjob-name}-node-0-0 -f | grep "loss"
   ```

4. **训练完成**：TrainJob 状态变为 Completed，检查 checkpoint 输出
   ```bash
   kubectl get trainjob llamafactory-lora-sft-demo
   ```

### Phase 1 涉及的文件清单

| 文件路径 | 操作 | 说明 |
|----------|------|------|
| `cmd/trainers/llamafactory/Dockerfile` | **新增** | LLaMA Factory 训练镜像 |
| `cmd/trainers/llamafactory/entrypoint.sh` | **新增** | PET_ 环境变量桥接脚本 |
| `manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml` | **新增** | ClusterTrainingRuntime 定义 |
| `manifests/base/runtimes/llamafactory/kustomization.yaml` | **新增** | Kustomize 注册 |
| kustomization.yaml | **修改** | 添加 `llamafactory` 到 resources 列表 |
| `examples/llamafactory/lora-sft-configmap.yaml` | **新增** | 训练配置 ConfigMap 示例 |
| `examples/llamafactory/lora-sft-trainjob.yaml` | **新增** | TrainJob 提交示例 |

---

## Phase 2：BuiltinTrainer 深度集成

**目标**：参考 TorchTune 的集成模式，将 LLaMA Factory 作为 Kubeflow Trainer 的 BuiltinTrainer，提供：
- Torch Plugin 中的 LLaMA Factory 命令变换和参数验证
- 多种预置 ClusterTrainingRuntime（LoRA SFT、Full SFT、DPO）
- Kubeflow Python SDK 中的 `LlamaFactoryConfig` 支持

### Step 1：添加常量定义

**文件**：constants.go

参考 TorchTune 的常量定义模式（搜索 `TorchTune` 相关常量），添加 LLaMA Factory 相关常量：

```go
// LLaMA Factory Entrypoint
LlamaFactoryEntrypoint = []string{"llamafactory-cli", "train"}

// LLaMA Factory 训练阶段
LlamaFactoryStageSFT  = "sft"
LlamaFactoryStageDPO  = "dpo"
LlamaFactoryStageRM   = "rm"

// LLaMA Factory 微调类型
LlamaFactoryFinetuningTypeFull = "full"
LlamaFactoryFinetuningTypeLora = "lora"
LlamaFactoryFinetuningTypeQlora = "qlora"

// LLaMA Factory 不可变配置（从 Runtime 中提取传递）
LlamaFactoryModelPath  = "model_name_or_path"
LlamaFactoryDatasetDir = "dataset_dir"
LlamaFactoryOutputDir  = "output_dir"

LlamaFactoryImmutableConfigs = sets.New(
    LlamaFactoryModelPath,
    LlamaFactoryDatasetDir,
    LlamaFactoryOutputDir,
)

// LLaMA Factory 需要桥接的 PET_ 环境变量映射
LlamaFactoryEnvBridge = map[string]string{
    TorchEnvNumNodes:       "NNODES",
    TorchEnvNumProcPerNode: "NPROC_PER_NODE",
    TorchEnvNodeRank:       "NODE_RANK",
    TorchEnvMasterAddr:     "MASTER_ADDR",
    TorchEnvMasterPort:     "MASTER_PORT",
}
```

> **参考**：constants.go 中搜索 `TorchTune` 查看所有 TorchTune 常量的定义模式。

### Step 2：扩展 Torch Plugin

**文件**：`pkg/runtime/framework/plugins/torch/llamafactory.go`（新增）

参考 `torchtune.go` 的实现模式，创建 LLaMA Factory 专属逻辑：

```go
// 核心函数签名（参考 torchtune.go 中的同名模式）

// validateLlamaFactory 验证 TrainJob 中的 LLaMA Factory 配置
// - 检查入口命令是否为 LlamaFactoryEntrypoint
// - 验证训练配置文件路径是否存在于 args 中
func validateLlamaFactory(trainJob *trainer.TrainJob, info *runtime.Info) error

// enforceLlamaFactoryPolicy 在 EnforceMLPolicy 中被调用
// - 注入桥接环境变量（PET_NNODES → NNODES 等）
// - 设置 FORCE_TORCHRUN=1
// - 提取 ConfigMap 中的不可变配置参数
func enforceLlamaFactoryPolicy(trainJob *trainer.TrainJob, info *runtime.Info) error
```

**文件**：torch.go（修改）

在 `Validate()` 和 `EnforceMLPolicy()` 中添加 LLaMA Factory 入口点检测分支：

```go
// 在 Validate() 中：
if slices.Equal(trainJob.Spec.Trainer.Command, constants.LlamaFactoryEntrypoint) {
    return validateLlamaFactory(trainJob, info)
}

// 在 EnforceMLPolicy() 中，PET_ 变量注入之后：
if slices.Equal(trainJob.Spec.Trainer.Command, constants.LlamaFactoryEntrypoint) {
    return enforceLlamaFactoryPolicy(trainJob, info)
}
```

**Plugin 的核心行为**——环境变量桥接注入：

与 Phase 1 中 shell 脚本桥接不同，Phase 2 在 Go 代码层面直接注入双重环境变量，消除对 entrypoint.sh 的依赖：

```go
// 在 enforceLlamaFactoryPolicy 中：
// 除了已有的 PET_* 变量外，额外注入 LLaMA Factory 需要的无前缀变量
for petName, lfName := range constants.LlamaFactoryEnvBridge {
    // 找到已注入的 PET_ 变量值，创建同值的无前缀变量
    for _, env := range trainerContainer.Env {
        if env.Name == petName {
            apply.UpsertEnvVars(&trainerContainer.Env,
                corev1ac.EnvVar().WithName(lfName).WithValue(env.Value),
                // 或 WithValueFrom(env.ValueFrom) 对于使用 fieldRef 的变量
            )
        }
    }
}

// 注入 FORCE_TORCHRUN=1
apply.UpsertEnvVars(&trainerContainer.Env,
    corev1ac.EnvVar().WithName("FORCE_TORCHRUN").WithValue("1"),
)
```

> **参考**：torch.go 第 126-165 行，TorchTune 的入口点检测和命令变换逻辑。

### Step 3：创建 Dockerfile 和 Requirements

**文件**：`cmd/trainers/llamafactory/Dockerfile`（简化版，不再需要 entrypoint.sh）

```dockerfile
FROM docker.io/hiyouga/llamafactory:latest
WORKDIR /workspace
```

Phase 2 不再需要 `entrypoint.sh`，因为环境变量桥接由 Torch Plugin 在控制器层面完成。

### Step 4：创建多种 ClusterTrainingRuntime

**目录结构**：
```
manifests/base/runtimes/llamafactory/
├── kustomization.yaml
├── llamafactory_lora_sft.yaml      # LoRA SFT（最常用）
├── llamafactory_full_sft.yaml      # Full SFT 无 DeepSpeed（小模型）
└── llamafactory_full_sft_ds.yaml   # Full SFT + DeepSpeed ZeRO-3（大模型）
```

**`llamafactory_full_sft_ds.yaml` 示例**（附带 DeepSpeed 配置）：

额外挂载 DeepSpeed 配置的 ConfigMap：
```yaml
# 在 node 容器中增加 volumes：
volumes:
  - name: deepspeed-config
    configMap:
      name: llamafactory-ds-z3-config
```

训练配置 YAML 中指定 DeepSpeed：
```yaml
deepspeed: /config/ds_z3_config.json
finetuning_type: full
```

### Step 5：SDK 集成

**代码库**：[kubeflow/sdk](https://github.com/kubeflow/sdk)（独立仓库）

参考 `TorchTuneConfig` 的实现，添加 `LlamaFactoryConfig`：

```python
@dataclass
class LlamaFactoryConfig:
    """Configuration for LLaMA Factory BuiltinTrainer."""
    
    # 训练阶段：sft, dpo, rm
    stage: str = "sft"
    
    # 微调类型：lora, full, qlora
    finetuning_type: str = "lora"
    
    # LoRA 参数
    lora_rank: int = 8
    lora_target: str = "all"
    
    # 训练超参数
    learning_rate: float = 1e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    cutoff_len: int = 1024
    bf16: bool = True
    
    # 资源配置
    resources_per_node: Optional[Dict[str, str]] = None
```

**SDK 使用示例**：
```python
from kubeflow.training import TrainingClient, Initializer
from kubeflow.training import HuggingFaceDatasetInitializer, HuggingFaceModelInitializer
from kubeflow.training import BuiltinTrainer, LlamaFactoryConfig

client = TrainingClient()
job_name = client.train(
    runtime="llamafactory-lora-sft",
    initializer=Initializer(
        dataset=HuggingFaceDatasetInitializer(storage_uri="hf://tatsu-lab/alpaca"),
        model=HuggingFaceModelInitializer(
            storage_uri="hf://Qwen/Qwen2.5-1.5B",
            access_token=os.environ["HF_TOKEN"],
        ),
    ),
    trainer=BuiltinTrainer(
        config=LlamaFactoryConfig(
            stage="sft",
            finetuning_type="lora",
            lora_rank=16,
            resources_per_node={"gpu": 1},
        ),
    ),
)
```

### Step 6：添加测试

**单元测试**（参考 torch_test.go）：

**文件**：`pkg/runtime/framework/plugins/torch/llamafactory_test.go`

```go
// 测试用例：
// - TestValidateLlamaFactory：验证 LLaMA Factory 入口点识别
// - TestEnforceLlamaFactoryPolicy：验证环境变量桥接注入
// - TestLlamaFactoryEnvBridge：验证 PET_* → LLaMA Factory 变量映射完整性
```

**集成测试**（参考 integration）：

**文件**：`test/integration/controller/llamafactory_test.go`

```go
// 测试用例：
// - TrainJob 引用 llamafactory-lora-sft runtime 后，生成的 JobSet 结构正确
// - Node Pod 包含正确的环境变量（PET_* 和桥接后的无前缀变量）
// - FORCE_TORCHRUN=1 被正确注入
```

### Phase 2 涉及的文件清单

| 文件路径 | 操作 | 说明 |
|----------|------|------|
| constants.go | **修改** | 添加 LLaMA Factory 常量 |
| `pkg/runtime/framework/plugins/torch/llamafactory.go` | **新增** | LLaMA Factory Plugin 逻辑 |
| `pkg/runtime/framework/plugins/torch/llamafactory_test.go` | **新增** | Plugin 单元测试 |
| torch.go | **修改** | 添加 LLaMA Factory 入口检测分支 |
| `cmd/trainers/llamafactory/Dockerfile` | **修改** | 简化（移除 entrypoint.sh） |
| `cmd/trainers/llamafactory/entrypoint.sh` | **删除** | 不再需要（Plugin 层完成桥接） |
| `manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml` | **存在** | Phase 1 已创建 |
| `manifests/base/runtimes/llamafactory/llamafactory_full_sft.yaml` | **新增** | Full SFT runtime |
| `manifests/base/runtimes/llamafactory/llamafactory_full_sft_ds.yaml` | **新增** | Full SFT + DeepSpeed runtime |
| `test/integration/controller/llamafactory_test.go` | **新增** | 集成测试 |

---

## 数据流架构图

```
用户提交 TrainJob
    │
    ▼
Kubeflow Trainer Controller
    │
    ├─ 读取 ClusterTrainingRuntime "llamafactory-lora-sft"
    ├─ 运行 Torch Plugin (EnforceMLPolicy)
    │   ├─ 注入 PET_NNODES, PET_MASTER_ADDR 等环境变量
    │   └─ [Phase 2] 注入桥接变量 NNODES, MASTER_ADDR 等 + FORCE_TORCHRUN=1
    ├─ 创建 JobSet
    │   ├─ Job: dataset-initializer  → 下载数据到 PVC /workspace/dataset
    │   ├─ Job: model-initializer    → 下载模型到 PVC /workspace/model
    │   └─ Job: node (replicas=numNodes)
    │       ├─ Pod 0 (NODE_RANK=0): llamafactory-cli train /config/training.yaml
    │       │   └─ torchrun → 训练进程 (master)
    │       └─ Pod 1 (NODE_RANK=1): llamafactory-cli train /config/training.yaml
    │           └─ torchrun → 训练进程 (worker)
    │
    ▼
训练完成 → checkpoint 保存到 PVC /workspace/output
```

---

## 关键参考文件索引

| 参考文件 | 说明 |
|----------|------|
| llama3_2_1B.yaml | TorchTune Runtime YAML 模板 |
| torchtune.go | TorchTune Plugin 实现 |
| torch.go | Torch Plugin 核心（PET_ 变量注入，第 126-165 行） |
| constants.go | 常量定义（TorchTune 常量模式，第 177-302 行） |
| Dockerfile | TorchTune 镜像构建参考 |
| launcher.py | LLaMA Factory 分布式启动逻辑 |
| Dockerfile | LLaMA Factory 官方 Docker 镜像 |
| qwen3_lora_sft.yaml | LLaMA Factory LoRA SFT 配置示例 |
| dataset_info.json | LLaMA Factory 数据集注册格式 |

---

## 后续扩展方向

1. **DPO/RLHF 支持**：添加 `llamafactory-dpo` ClusterTrainingRuntime，训练配置中改为 `stage: dpo`
2. **DeepSpeed ZeRO 大模型支持**：已在 Phase 2 的 `llamafactory_full_sft_ds.yaml` 中预留
3. **FSDP 支持**：通过 Accelerate 配置文件传入，使用 Torch Runtime 即可
4. **多模态训练**：LLaMA Factory 支持 VLM 训练，可增加对应的 ClusterTrainingRuntime
5. **Elastic Training**：利用 LLaMA Factory 的 `RDZV_ID` 参数支持弹性训练
