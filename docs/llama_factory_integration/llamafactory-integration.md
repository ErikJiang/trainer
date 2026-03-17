# LLaMA Factory on Kubeflow Trainer 集成设计方案

## 概述

本文档描述如何将 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 微调框架逐步落到 Kubernetes / Kubeflow Trainer 环境中运行，分为两个阶段：

- **Phase 1**：使用纯 Kubernetes Job（纯 YAML，不修改 Go 代码），先验证单节点 LoRA SFT 的最小可运行路径
- **Phase 2**：作为 BuiltinTrainer 深度集成（修改 Go 代码 + SDK），提供一等公民体验

**核心可行性依据**：LLaMA Factory 使用标准 `torchrun` 进行分布式训练，与 Kubeflow Trainer 的 Torch Plugin 协议层完全对齐。Kubeflow Trainer ROADMAP [issue #2752](https://github.com/kubeflow/trainer/issues/2752) 已将 LLaMA Factory 列为待探索的微调库。

**参考实现**：TorchTune 已完整集成到 Kubeflow Trainer 中（Dockerfile、ClusterTrainingRuntime、Torch Plugin、SDK），是本方案的核心参考模板。

---

## 关键设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| LLaMA Factory 版本 | **Legacy**（非 V1） | 更稳定、100+ 模型支持、文档更丰富，分布式协议与 V1 一致 |
| 训练阶段 | **LoRA SFT** | 最主流微调方式，显存需求低，适合小资源验证 |
| 分布式模式 | **单节点 torchrun**（Phase 1） | 先验证最小闭环，再考虑多节点 DDP |
| Runtime 类型 | **Kubernetes Job**（Phase 1） | 不引入 TrainJob / JobSet / Torch Plugin 额外复杂度 |
| 配置传递 | **ConfigMap → YAML 文件** | 最符合 LLaMA Factory 使用习惯 |

---

## 启动方式与 PET_ 结论

### 当前示例为什么不需要 PET_ 环境变量桥接

当前仓库中的示例已经切换为**单节点 Kubernetes Job + 显式 torchrun**，因此不再依赖 Kubeflow Torch Plugin 注入的 `PET_*` 环境变量。

直接原因有两点：

- 单节点场景直接在命令行里传入 `--standalone --nnodes=1 --nproc_per_node=1`，不需要 rendezvous 相关变量。
- 训练入口直接执行 `torchrun`，不再通过 `llamafactory-cli` 读取 `NNODES`、`MASTER_ADDR` 这组环境变量后再二次拉起。

当前示例的启动命令如下：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m llamafactory.launcher /config/training.yaml
```

因此，针对你关心的第 3 点，结论可以直接写成：

- **如果示例采用单节点 Kubernetes Job，并且由容器显式执行 torchrun，那么不需要做 PET_ 环境变量桥接。**
- **只有在回到 TrainJob + Torch Plugin + LLaMA Factory launcher 这条路径时，才需要处理 PET_* 与标准 torchrun 变量名不一致的问题。**

### 为什么不再使用 llamafactory-cli

`llamafactory-cli train` 本质上会在满足条件时再次调用 `torchrun`。在单节点验证场景里，这一层包装没有带来额外价值，反而会把启动逻辑拆成两段：

```bash
llamafactory-cli train /config/training.yaml
  -> launcher.py 读取环境变量
  -> 再次调用 torchrun
```

直接执行 `torchrun` 更干净，启动链路也更容易排障：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m llamafactory.launcher /config/training.yaml
```

这里仍然是 LLaMA Factory 的训练逻辑在运行，只是去掉了 `llamafactory-cli` 这一层额外封装。

### LLaMA Factory 的 torchrun 启动流程

如果仍通过 `llamafactory-cli` 进入，launcher 的行为如下：

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

这也是当前示例改为显式 `torchrun` 的原因：对单节点场景没有必要先走 `llamafactory-cli`，再由它转一次 `torchrun`。

---

## Phase 1：单节点 Job 快速验证

**目标**：不修改任何 Go 代码，纯通过 Docker 镜像 + YAML 配置，在 Kubernetes 上跑通单节点 LLaMA Factory LoRA SFT。

**验证环境**：1 节点 × 1 GPU，小模型（Qwen3-0.6B）

### Step 1：创建单文件 Demo Manifest

**文件**：`examples/llamafactory/lora-sft-demo.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llamafactory-lora-sft-demo
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 30Gi
  volumeMode: Filesystem
---
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
    lora_alpha: 16
    lora_dropout: 0.05
    lora_target: q_proj,v_proj

    ### dataset
    dataset_dir: /config
    dataset: alpaca_en_demo
    template: qwen3_nothink
    cutoff_len: 1024
    max_samples: 500
    val_size: 0.1
    preprocessing_num_workers: 4

    ### output
    output_dir: /workspace/output
    logging_steps: 5
    save_steps: 20
    save_total_limit: 2
    plot_loss: true
    overwrite_output_dir: true

    ### train
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    learning_rate: 1.0e-4
    num_train_epochs: 1.0
    do_eval: true
    per_device_eval_batch_size: 2
    eval_strategy: steps
    eval_steps: 20
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    seed: 42
    report_to: none
    bf16: true

  dataset_info.json: |
    {
      "alpaca_en_demo": {
        "file_name": "/workspace/dataset/data"
      }
    }
---
apiVersion: batch/v1
kind: Job
metadata:
  name: llamafactory-lora-sft-demo
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      initContainers:
      - name: dataset-initializer
        image: ghcr.io/kubeflow/trainer/dataset-initializer
        env:
        - name: HF_ENDPOINT
          value: https://hf-mirror.com
        - name: STORAGE_URI
          value: hf://tatsu-lab/alpaca
        volumeMounts:
        - mountPath: /workspace
          name: workspace
      - name: model-initializer
        image: ghcr.io/kubeflow/trainer/model-initializer
        env:
        - name: HF_ENDPOINT
          value: https://hf-mirror.com
        - name: STORAGE_URI
          value: hf://Qwen/Qwen3-0.6B
        volumeMounts:
        - mountPath: /workspace
          name: workspace
      containers:
      - name: trainer
        image: docker.io/hiyouga/llamafactory:0.9.4
        command:
        - /bin/sh
        - -c
        - |
          set -eu
          exec torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=1 \
            -m llamafactory.launcher \
            /config/training.yaml
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - mountPath: /workspace
          name: workspace
        - mountPath: /config
          name: training-config
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: llamafactory-lora-sft-demo
      - name: training-config
        configMap:
          name: llamafactory-lora-sft-config
```

> **关键配置说明**：
> - `initContainers` 顺序完成数据集和模型下载，主训练容器只有在两者完成后才会启动
> - `torchrun --standalone --nnodes=1 --nproc_per_node=1`：显式使用 torchrun，但限定为单节点场景
> - `dataset_dir: /config`：让 LLaMA Factory 从 `/config/dataset_info.json` 读取数据集注册信息，`file_name` 再指向 `/workspace/dataset/data`
> - 当前示例不依赖 `PET_*` 变量，因此没有额外桥接脚本

**提交命令**：
```bash
kubectl apply -f examples/llamafactory/lora-sft-demo.yaml
```

### Step 2：验证

**验证清单**：

1. **Pod 启动顺序**：dataset-initializer 和 model-initializer 先完成 → trainer 容器启动
   ```bash
   kubectl get pod -l app.kubernetes.io/name=llamafactory-lora-sft-demo -w
   ```

2. **torchrun 启动**：检查 trainer 容器日志中的 torchrun 启动信息
   ```bash
   kubectl logs job/llamafactory-lora-sft-demo -c trainer | grep -E "torchrun|distributed|loss"
   ```

3. **训练进行中**：确认 loss 在下降
   ```bash
   kubectl logs job/llamafactory-lora-sft-demo -c trainer -f | grep "loss"
   ```

4. **训练完成**：Job 状态变为 `Complete`，检查 checkpoint 输出
   ```bash
   kubectl get job llamafactory-lora-sft-demo
   ```

### Phase 1 涉及的文件清单

| 文件路径 | 操作 | 说明 |
|----------|------|------|
| `examples/llamafactory/lora-sft-demo.yaml` | **更新** | 单文件 demo，包含 PVC、训练配置 ConfigMap 与单节点 Job |
| `docs/llama_factory_integration/run.md` | **更新** | 运行说明改为纯 Job 验证流程 |

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

与 Phase 1 中在 Runtime command 内桥接不同，Phase 2 在 Go 代码层面直接注入双重环境变量，消除对 shell 桥接逻辑的依赖：

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

**文件**：训练镜像 Dockerfile（可选简化版）

```dockerfile
FROM docker.io/hiyouga/llamafactory:latest
WORKDIR /workspace
```

Phase 2 不再需要 shell 级环境变量桥接，因为环境变量映射由 Torch Plugin 在控制器层面完成。

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
