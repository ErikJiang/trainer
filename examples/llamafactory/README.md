# LlamaFactory Prometheus Demo

这个目录包含一个最小可用的 LlamaFactory LoRA SFT Prometheus Remote Write 示例，目标是让训练过程中的 loss 和 eval_loss 直接从训练进程内上报到 Prometheus。

## 目录说明

- [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml)：Kubernetes Job 示例，包含 PVC、ConfigMap 和训练 Job。
- [examples/llamafactory/Dockerfile.prometheus](examples/llamafactory/Dockerfile.prometheus)：基于官方 LlamaFactory 镜像构建的 overlay 镜像。
- [examples/llamafactory/llamafactory_ext](examples/llamafactory/llamafactory_ext)：wrapper、callback 和 remote write 客户端实现。
- [examples/llamafactory/IMPLEMENTATION.md](examples/llamafactory/IMPLEMENTATION.md)：面向维护者的实现说明。

## 构建镜像

从仓库根目录执行：

```bash
docker build -f examples/llamafactory/Dockerfile.prometheus -t llamafactory-prometheus:latest .
```

如果你的集群节点无法直接使用本地镜像缓存，需要额外执行其中一种动作：

1. 将镜像推送到集群可访问的仓库，并同步修改 [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中的镜像地址。
2. 将镜像导入 kind、k3d 或其他本地集群节点。

## 需要修改的示例参数

提交 Job 之前，至少检查以下配置：

1. [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中 trainer 容器的 image。
2. [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中的 PROMETHEUS_REMOTE_WRITE_URL。
3. [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中的 PVC 名称、storageClassName 和 namespace。
4. [examples/llamafactory/lora-sft-demo.yaml](examples/llamafactory/lora-sft-demo.yaml) 中 ConfigMap 里的模型、数据集和训练超参数。

与指标上报直接相关的环境变量有：

- PROMETHEUS_REMOTE_WRITE_URL：Prometheus Remote Write 地址。
- TRAIN_RUN_UID：当前训练运行的唯一标识。示例默认从 Pod metadata.uid 注入。
- PROM_REMOTE_WRITE_JOB_NAME：训练任务名称标签。
- PROM_REMOTE_WRITE_NAMESPACE：训练命名空间标签。
- PROM_REMOTE_WRITE_TIMEOUT_SECONDS：remote write HTTP 请求超时。
- PROM_REMOTE_WRITE_MODEL_NAME：可选的模型标签。

## 提交示例

```bash
kubectl apply -f examples/llamafactory/lora-sft-demo.yaml
```

查看 Pod：

```bash
kubectl get pod -l app.kubernetes.io/name=llamafactory-lora-sft-demo
kubectl logs job/llamafactory-lora-sft-demo -c trainer -f
```

## 验证可行性

至少验证以下几点：

1. 训练日志中能看到 torchrun 正常启动，并开始输出 loss。
2. 输出目录中持续生成 /workspace/output/trainer_log.jsonl，说明 LlamaFactory 原生日志逻辑没有被破坏。
3. Prometheus 接收端可以查询到 llamafactory_train_loss、llamafactory_eval_loss、llamafactory_learning_rate 等时间序列。
4. 当将 PROMETHEUS_REMOTE_WRITE_URL 临时改成不可达地址时，训练仍继续执行，只会在日志中出现 warning。

建议查询时至少带上 run_uid 过滤，以便区分不同训练实例。

## 常见问题

### 为什么没有使用 sidecar 或 Pushgateway

这个示例有意避免新增组件，直接在训练进程内注册 callback，并使用 Prometheus Remote Write API 推送指标。这样依赖最小，部署最简单。

### 为什么没有直接修改上游 LlamaFactory 代码

这个目录采用 wrapper 注入 callback 的方式实现功能扩展，减少对上游源码的侵入，后续升级 LlamaFactory 基础镜像时也更容易维护。

### 为什么实现都放在 examples 目录

这里的目标是提供一个自包含 demo。把 Dockerfile、示例 YAML、overlay 代码和文档放在同一目录，可以降低理解和迁移成本。
