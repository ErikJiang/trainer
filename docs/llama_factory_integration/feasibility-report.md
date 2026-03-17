# Kubeflow Trainer 运行 LLaMA Factory 微调任务可行性调研报告

## 1. 调研背景与目标

Kubeflow Trainer 是 Kubeflow 生态中用于在 Kubernetes 上编排分布式训练任务的核心组件。目前已原生支持 TorchTune、DeepSpeed、MLX 等训练框架。[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 作为当前最流行的 LLM 微调框架之一，支持 100+ 模型、多种微调方法（LoRA / Full / QLoRA）以及多种训练阶段（SFT / DPO / RLHF），具有很高的集成价值。

本调研基于 Kubeflow Trainer v2.1.0（release-2.1 分支），先通过单节点 Kubernetes Job 方案（纯 YAML 配置，不修改 Go 代码）验证 LLaMA Factory LoRA SFT 的最小可运行路径，再为后续 Trainer 深度集成提供依据。

> Kubeflow Trainer ROADMAP [issue #2752](https://github.com/kubeflow/trainer/issues/2752) 已将 LLaMA Factory 列为待探索的微调库。

---

## 2. 方案概述

### 2.1 技术可行性依据

LLaMA Factory 底层使用标准 `torchrun` 进行训练。对单节点验证场景而言，直接使用 Kubernetes Job 调用 `torchrun --standalone` 就能跑通完整流程，不必先引入 TrainJob、ClusterTrainingRuntime 和 JobSet 这层编排复杂度。后续如果要扩展到 Kubeflow Trainer 多节点场景，再接入 Torch Plugin 即可。

### 2.2 关键设计选择

| 决策项 | 选择 | 理由 |
|--------|------|------|
| LLaMA Factory 版本 | Legacy（非 V1） | 更稳定，模型支持广泛，文档丰富 |
| 训练阶段 | LoRA SFT | 最主流微调方式，显存需求低，适合验证 |
| 分布式模式 | 单节点单进程 | 当前示例仅验证最小可运行路径，避免过早引入多节点复杂度 |
| Runtime 类型 | Kubernetes Job | 直接调用 `torchrun`，不依赖 Trainer CRD |
| 配置传递方式 | ConfigMap 挂载 YAML 文件 | 符合 LLaMA Factory 原生使用习惯 |
| 验证模型 | Qwen3-0.6B | 体积小（~1.3GB），便于快速验证 |

### 2.3 实现文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `examples/llamafactory/lora-sft-demo.yaml` | 更新 | 单文件 Demo（PVC + ConfigMap + Job） |
| `docs/llama_factory_integration/run.md` | 更新 | 运行说明改为纯 Job 流程 |
| `docs/llama_factory_integration/llamafactory-integration.md` | 更新 | 设计文档补充 torchrun 直起与 PET_ 结论 |

### 2.4 training.yaml 训练参数说明

为了让示例更容易理解，下面直接用“YAML 注释”的方式说明 `training.yaml` 里的参数含义，读者可以按顺序对照配置看。

```yaml
### model
model_name_or_path: /workspace/model   # 基础模型路径，这里读取 model-initializer 预下载到 PVC 中的模型
trust_remote_code: true               # 允许加载模型仓库中的自定义代码，Qwen 这类模型通常需要打开

### method
stage: sft                            # 训练阶段：监督微调 SFT
do_train: true                        # 是否真正执行训练
finetuning_type: lora                 # 微调方式：LoRA，只训练适配器参数，不改原模型权重
lora_rank: 8                          # LoRA 秩，越大可学习能力越强，但显存和参数量也会增加
lora_alpha: 16                        # LoRA 缩放系数，通常和 lora_rank 搭配设置
lora_dropout: 0.05                    # LoRA dropout，降低小数据集过拟合风险
lora_target: q_proj,v_proj            # 只对注意力层的 q/v 投影加 LoRA，属于常见轻量配置

### dataset
dataset_dir: /config                  # 数据集注册信息目录，LLaMA Factory 会在这里找 dataset_info.json
dataset: alpaca_en_demo               # 使用的数据集名称，需要和 dataset_info.json 里的 key 对应
template: qwen3_nothink               # 提示词模板，决定如何把样本拼成 Qwen3 训练输入
cutoff_len: 1024                      # 单条样本最大 token 长度，越大越占显存
max_samples: 500                      # 最多取 500 条样本，本例用于快速验证，不追求最佳效果
val_size: 0.1                         # 划出 10% 作为验证集
preprocessing_num_workers: 4          # 数据预处理并行 worker 数，提升 tokenization 速度

### output
output_dir: /workspace/output         # 输出目录，checkpoint、日志、loss 图都会写到这里
logging_steps: 5                      # 每 5 步打印一次训练日志
save_steps: 20                        # 每 20 步保存一次 checkpoint
save_total_limit: 2                   # 最多保留 2 个 checkpoint，避免输出目录占满
plot_loss: true                       # 训练后生成 loss 曲线，便于观察是否正常收敛
overwrite_output_dir: true            # 允许覆盖已有输出目录，方便反复跑 demo

### train
per_device_train_batch_size: 2        # 每张 GPU 的训练 batch size，直接影响显存占用
gradient_accumulation_steps: 4        # 梯度累积 4 步再更新一次参数，用小显存模拟更大 batch
learning_rate: 1.0e-4                 # 学习率，LoRA SFT 的常见起点
num_train_epochs: 1.0                 # 训练 1 个 epoch，适合可行性验证
do_eval: true                         # 是否开启验证集评估
per_device_eval_batch_size: 2         # 每张 GPU 的评估 batch size
eval_strategy: steps                  # 按训练步数触发评估
eval_steps: 20                        # 每 20 步评估一次
lr_scheduler_type: cosine             # 使用 cosine 学习率衰减策略
warmup_ratio: 0.1                     # 前 10% 步数做 warmup，减少训练初期不稳定
seed: 42                              # 随机种子，尽量保证结果可复现
report_to: none                       # 不上报到 WandB、TensorBoard 等外部实验平台
bf16: true                            # 使用 bfloat16 混合精度，降低显存占用并提升吞吐
ddp_timeout: 180000000                # DDP 初始化超时时间，避免集群启动较慢时误判失败
```

这份示例配置整体是“先跑通，再谈调优”的思路：

- `LoRA + Qwen3-0.6B`：先降低显存压力和训练复杂度
- `max_samples: 500`、`num_train_epochs: 1.0`：先缩短验证时间
- `per_device_train_batch_size: 2`、`gradient_accumulation_steps: 4`：在单卡显存受限时维持一个较稳妥的等效 batch
- `do_eval: true`、`save_steps: 20`、`plot_loss: true`：让训练过程和结果更容易观察

如果后续要从 demo 调到正式训练，通常优先关注这几类参数：

- 显存压力：`cutoff_len`、`per_device_train_batch_size`、`bf16`
- 训练效果：`learning_rate`、`num_train_epochs`、`lora_rank`、`lora_target`
- 数据规模：`max_samples`、`val_size`
- 工程策略：`save_steps`、`save_total_limit`、`report_to`

---

## 3. 核心问题与解决方案

### 3.1 是否还需要 PET_ 环境变量桥接（关键问题）

**结论**：当前单节点 Job 示例**不需要** PET_ 环境变量桥接。

原因很直接：

- 当前示例不再通过 TrainJob 进入 Kubeflow Torch Plugin，所以集群不会注入 `PET_*` 变量。
- 当前示例显式调用 `torchrun --standalone --nnodes=1 --nproc_per_node=1`，启动参数已经在命令行中给全了。
- 因为是单节点场景，也不需要 `MASTER_ADDR`、`MASTER_PORT`、`NODE_RANK` 这类多节点 rendezvous 参数。

只有在后续回到 **TrainJob + Torch Plugin + LLaMA Factory launcher** 这条路径时，才会重新遇到下面这类变量名不匹配问题：

```bash
PET_NNODES      -> NNODES
PET_NODE_RANK   -> NODE_RANK
PET_MASTER_ADDR -> MASTER_ADDR
```

那时可以选择两种做法之一：

- 在控制器或 entrypoint 中做 `PET_*` 到标准 torchrun 变量的桥接。
- 直接绕过 `llamafactory-cli`，由 runtime command 显式执行 `torchrun` 并传入标准参数。

### 3.2 数据集元信息注册

**问题描述**：LLaMA Factory 需要通过 `dataset_info.json` 注册数据集的路径和格式信息，而 Kubeflow 的 Dataset Initializer 下载数据后的目录结构（尤其是 HuggingFace 格式的 parquet 文件名包含随机哈希后缀）不可预知。

**解决方案**：
- 将 `dataset_info.json` 与训练配置一起放入 ConfigMap，挂载到 `/config` 目录
- `dataset_info.json` 中的 `file_name` 指向数据目录而非具体文件名（如 `/workspace/dataset/data`），LLaMA Factory 会自动发现目录下的所有同类型文件
- 训练配置中设置 `dataset_dir: /config` 指向 ConfigMap 挂载路径

### 3.3 存储共享方案

**问题描述**：训练流水线包含三个阶段（dataset-initializer → model-initializer → node），需要在不同阶段的 Pod 之间共享数据。

**解决方案**：使用独立的 PVC（`ReadWriteMany` 模式），所有阶段的 Pod 通过 `claimName` 挂载同一个 PVC。这样 Initializer 下载的模型和数据集可以被训练节点直接读取。

> 注意：当前实现使用了硬编码的 `storageClassName: nfs-client-test` 和固定的 `namespace: default`，这是环境特定的配置。

### 3.4 镜像可用性

**问题描述**：在中国大陆网络环境下，部分 Docker 镜像和 HuggingFace 模型下载受限。

**解决方案**：
- 训练镜像通过镜像加速站拉取（`docker.m.daocloud.io/hiyouga/llamafactory:latest`）
- HuggingFace 模型和数据集通过 `HF_ENDPOINT=https://hf-mirror.com` 环境变量走镜像站下载

---

## 4. 验证结果

基于上述方案，在单节点单 GPU 环境下可以完成以下验证：

1. **初始化流程正确**：dataset-initializer 和 model-initializer 作为 initContainer 顺序执行，训练容器仅在两者完成后启动
2. **torchrun 直起生效**：训练容器直接执行 `torchrun --standalone`，不存在 `llamafactory-cli` 二次拉起 torchrun 的链路
3. **LLaMA Factory LoRA SFT 训练可运行**：使用 Qwen3-0.6B 模型 + alpaca 数据集，训练 loss 正常下降，最终训练完成并输出 checkpoint

**结论：先用纯 Kubernetes Job 验证 LLaMA Factory 微调训练链路是合理且更稳妥的；在此基础上再演进到 Kubeflow Trainer 集成，边界会更清晰。**

---

## 5. 当前方案的已知局限

| 局限 | 说明 |
|------|------|
| 非声明式训练抽象 | 直接使用 Job，缺少 TrainJob 级别的状态抽象和统一管理 |
| 仍依赖 PVC | 模型、数据集和输出都挂在同一卷上，存储策略仍需用户自行规划 |
| 环境配置仍需手工调整 | `HF_ENDPOINT`、镜像源、PVC 容量等参数仍是示例级配置 |
| 缺乏参数校验 | YAML 中的训练参数没有控制器侧校验 |
| 无 SDK 支持 | 仍然只能通过 kubectl 操作 |
| 单一训练范式 | 仅验证了单节点 LoRA SFT |

---

## 6. 改进与优化方向

### 6.1 短期优化（Phase 1 完善）

- **参数化 PVC 和镜像源**：避免示例依赖固定存储和网络环境
- **补充多 GPU 单机示例**：把 `nproc_per_node` 调整为 GPU 数，验证单机多卡 torchrun 路径
- **为 Trainer 集成保留明确边界**：后续进入 TrainJob 场景时，再单独处理 PET_ 兼容与 runtime 模板
- **镜像版本锁定**：继续使用固定版本标签，保证复现性

### 6.2 中期目标（Phase 2 — BuiltinTrainer 深度集成）

参考 TorchTune 的集成模式，在 Go 代码层面实现原生支持：

- **Torch Plugin 扩展**：在 `pkg/runtime/framework/plugins/torch/` 中新增 `llamafactory.go`，由控制器自动完成 PET_ → LLaMA Factory 环境变量桥接，消除 shell 脚本依赖
- **常量与校验**：在 `pkg/constants/` 中定义 LLaMA Factory 相关常量，在 Plugin 的 `Validate()` 中加入参数合法性校验
- **多训练范式 Runtime**：预置 LoRA SFT、Full SFT、Full SFT + DeepSpeed ZeRO-3 等多种 ClusterTrainingRuntime
- **Python SDK 支持**：在 Kubeflow SDK 中实现 `LlamaFactoryConfig` 数据类，支持编程方式创建训练任务

### 6.3 远期展望

- **DPO / RLHF 支持**：添加对齐训练阶段的 Runtime 模板
- **DeepSpeed ZeRO 大模型训练**：支持 ZeRO-2/3 模型分片，解锁 7B+ 模型的 Full SFT
- **FSDP 支持**：通过 Accelerate 配置文件传入，利用 Torch Runtime 原生支持
- **多模态训练**：LLaMA Factory 支持 VLM（视觉语言模型）训练，可新增对应 Runtime
- **弹性训练（Elastic Training）**：结合 LLaMA Factory 的 RDZV 参数支持节点动态伸缩
- **Checkpoint 自动推送**：训练完成后自动将 LoRA adapter 或合并模型推送至 HuggingFace Hub 或 S3

---

## 7. 总结

本次调研先通过单节点 Kubernetes Job 的纯 YAML 方案，验证了 LLaMA Factory LoRA SFT 的最小可运行路径。当前示例直接使用 `torchrun --standalone` 启动训练，不再依赖 `llamafactory-cli` 间接拉起训练进程，也不需要处理 `PET_*` 环境变量桥接。

这条路径的价值在于先把镜像、数据集初始化、模型下载和训练配置本身跑通，再决定是否把它上收进 Kubeflow Trainer 的 TrainJob/BuiltinTrainer 抽象。后续如果进入多节点 Trainer 集成，再单独处理 Torch Plugin 注入的 `PET_*` 变量与 LLaMA Factory 所需标准 torchrun 参数之间的兼容问题，会更清晰也更可控。
