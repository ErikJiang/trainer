# LlamaFactory Integration — 开发记忆总结

> 最后更新: 2026-03-19 | 分支: `llamafactory-phase2` (trainer), `llamafactory-sdk` (sdk)

---

## 一、项目概况

在 Kubeflow Trainer 中集成 LlamaFactory 训练框架，分两阶段：
- **Phase 1 (Go 控制器侧)**: 修复 Bug + 补齐测试 + ipynb 示例 → trainer 仓库
- **Phase 2 (Python SDK 侧)**: 新增 `LlamaFactoryConfig` 类型 + ConfigMap 自动创建 → sdk 仓库

两个阶段均已完成并推送。

---

## 二、仓库与分支

| 仓库 | 远程地址 | 分支 | 状态 |
|------|----------|------|------|
| trainer | `https://github.com/ErikJiang/trainer.git` | `llamafactory-phase2` | ✅ 已推送 |
| sdk | `https://github.com/ErikJiang/sdk.git` | `llamafactory-sdk` | ✅ 已推送 |

- trainer 本地路径: `D:\Workspace\projects\ErikJiang\trainer`
- sdk 本地路径: `D:\Workspace\projects\ErikJiang\trainer\sdk-repo`
- sdk WSL Python 环境: `sdk-repo/.venv` (Python 3.12.3)

---

## 三、Phase 1 — Go 控制器改动

### 3.1 P0 Bug 修复: Entrypoint 检测逻辑

**问题**: `torch.go` 的 `EnforceMLPolicy` 和 `Validate` 用 `trainJob.Spec.Trainer.Command` 判断 entrypoint，但用户 TrainJob 不设 Command 时该字段为 nil，导致 LlamaFactory 走错分支（按 torchrun 配置）。

**修复**: 构建 `effectiveCommand` — 优先取 `trainJob.Spec.Trainer.Command`，fallback 到 Info 对象中 `trainerContainer.Command`（即 runtime 定义的 command）。

**修改文件**:
- `pkg/runtime/framework/plugins/torch/torch.go` — `EnforceMLPolicy()` 和 `Validate()` 方法

### 3.2 测试补充

- `pkg/runtime/framework/plugins/torch/llamafactory_test.go` — 新增 nil-command 场景的单元测试
- `test/integration/controller/core/trainingruntime_test.go` — 新增 LlamaFactory 集成测试用例

### 3.3 ipynb 示例

- `examples/llamafactory/lora-sft.ipynb` — 使用 SDK `LlamaFactoryConfig` 的完整训练流程 notebook

### 3.4 验证命令

```bash
make test                    # Go 单元测试
make golangci-lint           # lint 检查
make test-integration        # 集成测试 (需要环境)
```

---

## 四、Phase 2 — SDK 改动

### 4.1 设计决策

采用 **方案 A: 纯 dict 透传**。LlamaFactory 有 500+ 参数，逐一映射不现实。`LlamaFactoryConfig.config: dict[str, Any]` 直接对应 LlamaFactory 的 YAML 配置，零版本耦合。

### 4.2 改动文件清单

| 文件 | 改动说明 |
|------|----------|
| `types/types.py` | 新增 `LlamaFactoryConfig` dataclass (`config: dict[str, Any]`, `num_nodes`, `resources_per_node`) |
| `constants/constants.py` | 新增 `LLAMA_FACTORY_COMMAND`, `LLAMA_FACTORY_CONFIG_PATH`, `LLAMA_FACTORY_CONFIG_MAP_KEY` |
| `backends/kubernetes/utils.py` | 新增 `_get_trainer_cr_from_llamafactory_config()`, `get_configmap_for_llamafactory()`; 更新 `get_runtime_trainer()` 和 `get_trainer_cr_from_builtin_trainer()` |
| `backends/kubernetes/backend.py` | `train()` 中为 LlamaFactory 自动创建 ConfigMap (`{trainjob-name}-config`) |
| `__init__.py` | 导出 `LlamaFactoryConfig` |
| `backends/kubernetes/utils_test.py` | 4 个新测试 (ConfigMap 生成 + trainer CR 构建) |
| `backends/kubernetes/backend_test.py` | 新增 LlamaFactory test_train 用例 |

### 4.3 ConfigMap 自动化流程

```
LlamaFactoryConfig.config (dict)
  → yaml.safe_dump() 序列化
  → 构建 ConfigMap (name: {trainjob}-config, key: training.yaml)
  → backend.train() 中 core_api.create_namespaced_config_map()
  → 然后创建 TrainJob
```

### 4.4 验证命令

```bash
cd sdk-repo
wsl -- bash -c "cd /mnt/d/.../sdk-repo && source .venv/bin/activate && python -m pytest kubeflow/trainer/backends/kubernetes/utils_test.py -v"
# 结果: 38 passed (含 4 个新 LlamaFactory 测试)
```

### 4.5 已知问题

- `backend_test.py::test_train` 存在 **预存错误**: `kubeflow-trainer-api==2.1.0` 缺少 `TrainerV1alpha1RuntimePatch` 模型，导致导入失败。**非本次变更引起**，需要升级 `kubeflow-trainer-api` 包解决。

---

## 五、LlamaFactory 架构要点

### 与 TorchTune 的核心差异

| 维度 | TorchTune | LlamaFactory |
|------|-----------|-------------|
| 配置方式 | CLI args (`tune run ...`) | YAML 配置文件 (`llamafactory-cli train /config/training.yaml`) |
| SDK 映射 | `TorchTuneConfig` → args list | `LlamaFactoryConfig` → ConfigMap YAML |
| Entrypoint | `tune`, `tune run` | `llamafactory-cli`, `train` |
| 容器镜像 | `ghcr.io/kubeflow/trainer/torchtune-trainer` | `ghcr.io/kubeflow/trainer/llamafactory-trainer` |
| Config 存储 | 无需额外资源 | ConfigMap 挂载至 `/config/training.yaml` |

### Runtime 结构

- ClusterTrainingRuntime: `manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml`
- 默认模型: Qwen/Qwen2.5-1.5B-Instruct
- 默认数据集: tatsu-lab/alpaca
- Volume 挂载: `/workspace` (PVC for model+dataset), `/config` (ConfigMap for training config)

### Entrypoint 检测 (torch plugin)

`torch.go` 中通过 `slices.Equal(effectiveCommand, llamaFactoryEntrypoint)` 判断是否走 LlamaFactory 分支：
- 是 → 调用 `enforceLlamaFactoryPolicy()` 设置环境变量桥接
- 否 → 走 torchrun 标准流程

---

## 六、关键文件速查

### Trainer 仓库
- Runtime 定义: `manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml`
- Torch plugin: `pkg/runtime/framework/plugins/torch/torch.go`
- LlamaFactory 测试: `pkg/runtime/framework/plugins/torch/llamafactory_test.go`
- 集成测试: `test/integration/controller/core/trainingruntime_test.go`
- YAML 示例: `examples/llamafactory/lora-sft-configmap.yaml`, `lora-sft-trainjob.yaml`
- Notebook 示例: `examples/llamafactory/lora-sft.ipynb`
- Dockerfile: `cmd/trainers/llamafactory/Dockerfile`

### SDK 仓库 (sdk-repo/)
- 类型: `kubeflow/trainer/types/types.py`
- 常量: `kubeflow/trainer/constants/constants.py`
- 工具函数: `kubeflow/trainer/backends/kubernetes/utils.py`
- 后端: `kubeflow/trainer/backends/kubernetes/backend.py`
- 测试: `kubeflow/trainer/backends/kubernetes/utils_test.py`, `backend_test.py`

---

## 七、后续可扩展方向

1. **更多 Runtime**: 可新增 full-finetune、DPO 等 ClusterTrainingRuntime
2. **ConfigMap 清理**: 当前 ConfigMap 创建后无自动清理机制，可考虑 OwnerReference 绑定 TrainJob
3. **dataset_info.json**: 当前 ipynb 示例未包含 dataset_info.json 生成逻辑，复杂场景可能需要
4. **backend_test.py 修复**: 等 `kubeflow-trainer-api` 升级后验证 LlamaFactory backend 测试
5. **ARM64 支持**: 受上游 LlamaFactory 镜像限制，目前仅支持 linux/amd64
