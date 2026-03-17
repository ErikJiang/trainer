# LlamaFactory LoRA SFT Prometheus Remote Write 集成设计

## 概述

本文档描述如何在当前的 LlamaFactory LoRA SFT Kubernetes Job 示例基础上，为训练过程中的 loss 和 eval_loss 增加 Prometheus Remote Write 上报能力。

目标约束如下：

- 保留当前 torchrun 启动方式
- 不引入 Pushgateway、sidecar exporter 或其他新组件
- 尽量不维护 LlamaFactory 上游源码分叉
- 优先采用最小依赖的自定义镜像 overlay 方式
- 训练过程中的指标上报失败不能影响训练本身

推荐方案为：

在当前 LlamaFactory 镜像之上增加一个很薄的 Python overlay，提供一个自定义 wrapper 入口和一个自定义 TrainerCallback。Job 仍然使用 torchrun 启动，但模块入口从上游的 llamafactory.launcher 切换为自定义 entrypoint。该 entrypoint 内部调用 LlamaFactory 的 run_exp，并注入 Prometheus Remote Write callback，在训练进程内直接从 TrainerState.log_history 读取 loss、eval_loss、learning_rate 等指标并主动上报。

这条路径本质上复用了 datatunerx 的设计思路，但不直接复用 datatunerx 的训练代码。

## 现状分析

### 当前 demo 的运行方式

当前示例位于 [examples/llamafactory/lora-sft-demo.yaml](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/lora-sft-demo.yaml)。

它的特点是：

- 使用 batch/v1 Job 运行训练
- 使用 initContainer 下载数据和模型
- 主容器通过 torchrun 显式拉起 LlamaFactory
- 训练配置来自 ConfigMap 挂载的 training.yaml
- 当前没有任何 Prometheus 指标上报逻辑

当前方式已经适合做指标集成，因为它没有额外的调度层包装，入口清晰。

### datatunerx 中 loss 上报是如何实现的

datatunerx 的实现不是通过旁路脚本扫描日志，而是训练进程内 callback 主动上报。

关键链路如下：

1. 控制面注入 metrics_export_address 和 uid  
   参考 [datatunerx-repo/internal/controller/finetune/finetune_controller.go](/Users/jiang/workspace/Projects/erikjiang/trainer/datatunerx-repo/internal/controller/finetune/finetune_controller.go#L511-L512)

2. 训练参数中定义 metrics_export_address 和 uid  
   参考 [datatunerx-repo/cmd/tuning/parser.py](/Users/jiang/workspace/Projects/erikjiang/trainer/datatunerx-repo/cmd/tuning/parser.py#L202-L208)

3. 训练入口在创建 Trainer 时注册 LogCallback  
   参考 [datatunerx-repo/cmd/tuning/train.py](/Users/jiang/workspace/Projects/erikjiang/trainer/datatunerx-repo/cmd/tuning/train.py#L287-L295)

4. LogCallback 在 on_log 中从 state.log_history 读取 loss 和 eval_loss  
   参考 [datatunerx-repo/cmd/tuning/callback.py](/Users/jiang/workspace/Projects/erikjiang/trainer/datatunerx-repo/cmd/tuning/callback.py#L97-L155)

5. callback 调用 remote write 客户端将指标推送到 Prometheus  
   参考 [datatunerx-repo/cmd/tuning/prometheus/metrics.py](/Users/jiang/workspace/Projects/erikjiang/trainer/datatunerx-repo/cmd/tuning/prometheus/metrics.py#L16-L106)

结论：

- datatunerx 的核心能力来自 TrainerCallback
- 数据源是 TrainerState.log_history
- 上报方式是训练进程内主动 remote write
- 不是靠额外脚本 tail stdout 或轮询日志文件

### LlamaFactory 现有可复用能力

LlamaFactory 自身已经具备很适合复用的几个点：

1. run_exp 支持外部传入 callbacks  
   参考 [LlamaFactory-repo/src/llamafactory/train/tuner.py](/Users/jiang/workspace/Projects/erikjiang/trainer/LlamaFactory-repo/src/llamafactory/train/tuner.py#L107-L116)

2. 默认会追加自己的 LogCallback  
   参考 [LlamaFactory-repo/src/llamafactory/train/tuner.py](/Users/jiang/workspace/Projects/erikjiang/trainer/LlamaFactory-repo/src/llamafactory/train/tuner.py#L53-L65)

3. 默认 LogCallback 已经会在 on_log 时写 trainer_log.jsonl  
   参考 [LlamaFactory-repo/src/llamafactory/train/callbacks.py](/Users/jiang/workspace/Projects/erikjiang/trainer/LlamaFactory-repo/src/llamafactory/train/callbacks.py#L172-L280)

4. 参数解析已经支持从 YAML 直接读取  
   参考 [LlamaFactory-repo/src/llamafactory/hparams/parser.py](/Users/jiang/workspace/Projects/erikjiang/trainer/LlamaFactory-repo/src/llamafactory/hparams/parser.py#L69-L81)

这意味着我们不需要改写上游训练主流程，只需要在外层补一个入口和 callback。

## 方案选择

### 推荐方案

推荐方案是：

- 自定义镜像增加一个 wrapper 入口
- wrapper 调用 LlamaFactory 的 run_exp
- wrapper 注入自定义 Prometheus callback
- callback 在训练进程内直接 remote write 到 Prometheus

### 不推荐作为主方案的备选

备选方案是：

- 仍跑原始 torchrun
- 同时起一个 watcher 脚本
- watcher tail trainer_log.jsonl 或 stdout
- watcher 再 remote write

该方案不是不能做，但不建议作为主路径，原因如下：

- 需要处理双进程生命周期
- 需要处理文件追尾、去重、断点恢复
- 多卡场景更容易重复上报
- 本质上是在绕过 Trainer 原生 callback 注入点

如果已经接受自定义镜像，那么 wrapper 注入 callback 的方案更自然，也更稳。

## 文件级设计

### 1. 新增 wrapper 入口文件

已新增自定义 Python 模块 [examples/llamafactory/llamafactory_ext/train.py](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/llamafactory_ext/train.py)，作为新的 torchrun 模块入口。

建议职责：

- 读取 training.yaml 路径
- 读取环境变量中的 remote write 地址和 run id
- 构造 PrometheusRemoteWriteCallback
- 调用 LlamaFactory 的 run_exp，并通过 callbacks 参数注入 callback

建议模块职责边界：

- 不负责训练逻辑
- 不负责解析完整训练参数
- 不重写 LlamaFactory 的配置体系
- 只做入口拼装和 callback 注入

建议环境变量：

- PROMETHEUS_REMOTE_WRITE_URL
- TRAIN_RUN_UID
- PROM_REMOTE_WRITE_TIMEOUT_SECONDS
- PROM_REMOTE_WRITE_JOB_NAME
- PROM_REMOTE_WRITE_NAMESPACE

建议行为：

- 如果未设置 PROMETHEUS_REMOTE_WRITE_URL，则 callback 不启用
- 如果未设置 TRAIN_RUN_UID，则默认退回到 Job 名称或显式传入的名字
- wrapper 继续把原来的 training.yaml 路径传给 run_exp
- 保持与 torchrun 的兼容，不接管分布式控制逻辑

### 2. 新增 callback 文件

已新增 [examples/llamafactory/llamafactory_ext/callbacks.py](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/llamafactory_ext/callbacks.py)，其中实现 PrometheusRemoteWriteCallback。

建议职责：

- 在训练开始时初始化本地时间基线
- 在 on_log 中读取 state.log_history 最新一条
- 识别训练日志与评估日志
- 只在 rank 0 或 local process zero 上报
- 计算 current_steps、total_steps、elapsed_time、remaining_time
- 将结构化指标交给 remote write 客户端

建议采集字段：

- loss
- eval_loss
- learning_rate
- epoch
- current_steps
- total_steps
- percentage
- elapsed_time
- remaining_time

建议行为细节：

- on_train_begin 时记录 start_time 和 max_steps
- on_log 时读取 state.global_step 与 state.log_history
- eval 日志和 train 日志可以用不同 metric name，或统一 metric name 加 phase 标签
- 当字段缺失时跳过该指标，不抛异常
- remote write 失败只打印 warning，不中断训练

### 3. 新增 remote write 客户端文件

已新增 [examples/llamafactory/llamafactory_ext/remote_write.py](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/llamafactory_ext/remote_write.py)，提供轻量 remote write 客户端。

建议职责：

- 构造 TimeSeries
- 构造 WriteRequest
- snappy 压缩
- 向 Prometheus remote write 地址发起 POST 请求
- 做最小的超时和异常保护

建议接口：

- 一个 PrometheusRemoteWriter 类
- 一个发送单个指标的方法
- 一个发送一组指标的方法

建议请求策略：

- 默认短超时
- 异常吞掉并打印 warning
- 不做阻塞式重试
- 可选单线程异步池，但不是必须

重要设计决策：

不要沿用 datatunerx 当前把 loss 等数值塞进 label 的方式。  
推荐做法是：

- sample.value 承载 loss 或 learning_rate 数值
- label 只承载低基数字段

建议 label：

- run_uid
- phase
- job_name
- namespace
- model_name

建议 metric name：

- llamafactory_train_loss
- llamafactory_eval_loss
- llamafactory_learning_rate
- llamafactory_train_progress

这样更符合 Prometheus 的数据模型。

### 4. 新增 protobuf 生成文件

已新增 [examples/llamafactory/llamafactory_ext/prometheus_pb2.py](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/llamafactory_ext/prometheus_pb2.py) 作为静态 protobuf 生成文件。

理由：

- 避免在镜像构建时引入 protoc
- 只把它当作协议依赖物，不把它当作业务逻辑文件
- 保持构建链路简单

### 5. 自定义镜像设计

已新增镜像 overlay 文件 [examples/llamafactory/Dockerfile.prometheus](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/Dockerfile.prometheus)。

建议从仓库根目录构建：

```bash
docker build -f examples/llamafactory/Dockerfile.prometheus -t llamafactory-prometheus:latest .
```

如果集群节点不能直接访问本地 Docker 镜像缓存，还需要把该镜像导入到集群节点或推送到可访问的镜像仓库。当前示例中的 trainer 镜像已设置为 IfNotPresent，便于复用本地或节点本地缓存。

当前目录还新增了两份配套文档：

- [examples/llamafactory/README.md](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/README.md)
- [examples/llamafactory/IMPLEMENTATION.md](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/IMPLEMENTATION.md)

镜像增量应仅包括：

- 新增 wrapper 模块
- 新增 callback 模块
- 新增 remote write 客户端模块
- 新增 protobuf 生成文件
- 安装 requests
- 安装 protobuf
- 安装 python-snappy
- 设置 PYTHONPATH 让新增模块可被导入

不建议做的事情：

- 不 fork 上游 LlamaFactory 源码
- 不替换上游训练环境
- 不加入额外守护进程
- 不增加 sidecar

## Job YAML 改造设计

目标文件是 [examples/llamafactory/lora-sft-demo.yaml](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/lora-sft-demo.yaml)。

需要改动的点只有四类。

### 1. 替换训练镜像

将 trainer 容器镜像从官方 LlamaFactory 镜像替换为带 overlay 的自定义镜像。

### 2. 替换 torchrun 模块入口

当前入口已从上游的 llamafactory.launcher 改为自定义 wrapper 模块 llamafactory_ext.train。

保持以下内容不变：

- torchrun 仍然存在
- --standalone 不变
- --nnodes 不变
- --nproc_per_node 不变
- training.yaml 路径不变

也就是说，变的是模块入口，不是分布式启动模式。

### 3. 增加指标相关环境变量

建议至少注入：

- PROMETHEUS_REMOTE_WRITE_URL
- TRAIN_RUN_UID

可选注入：

- PROM_REMOTE_WRITE_TIMEOUT_SECONDS
- PROM_REMOTE_WRITE_JOB_NAME
- PROM_REMOTE_WRITE_NAMESPACE

关于 run uid：

- 如果这是 demo，最简单的是直接用 Job 名称作为 run_uid
- 如果后续需要更稳定的唯一性，再考虑由上层创建器显式模板化注入 uid

### 4. 保留现有 volume 和 config 结构

以下内容不需要改：

- workspace PVC
- /workspace 挂载
- /config 挂载
- training.yaml
- dataset_info.json
- 数据集和模型 initializer

这样可以把本次改造完全聚焦到“训练入口 + 指标上报”。

## 指标模型设计

建议不要完全照搬 datatunerx 当前的指标建模。

### 不推荐的做法

不推荐：

- 把 loss、eval_loss、learning_rate 放到 label 中

原因：

- 会造成高基数
- 查询不自然
- 违背 Prometheus 常见建模方式

### 推荐的做法

推荐把数值放入 sample.value，把标签维持在低基数。

建议分成以下序列：

- llamafactory_train_loss
- llamafactory_eval_loss
- llamafactory_learning_rate
- llamafactory_train_progress

建议公共 labels：

- run_uid
- phase
- job_name
- namespace

可选 labels：

- model_name
- stage

其中：

- loss 和 eval_loss 使用各自的 sample.value
- progress 可以用 current_steps 或 percentage 表达
- epoch 既可以作为单独指标，也可以作为 progress 辅助字段

## 多卡与容错设计

### 多卡去重

必须只允许 rank 0 或 local process zero 上报。  
否则在 nproc_per_node 大于 1 时会出现重复时间序列写入。

### 上报失败容错

Prometheus remote write 不可达时：

- 训练不能失败
- callback 只记录 warning
- 不抛出中断训练的异常

### 频率控制

上报频率直接受 logging_steps 和 eval_steps 影响。  
如果 logging_steps 过小，上报频率会偏高。建议在文档中提醒：

- demo 保持当前粒度即可
- 真正生产化时应避免每一步都 remote write

## 验收设计

### 单卡验收

验证点：

- 训练仍能正常启动
- 输出目录中的 trainer_log.jsonl 仍持续生成
- Prometheus 端可看到 train loss 和 eval loss 指标

### 多卡验收

验证点：

- 只产生一份指标流
- 没有同一步重复写入

### 容错验收

验证点：

- 将 remote write 地址改成不可达地址
- 训练继续进行
- 日志中仅出现 warning
- 训练不会因指标失败退出

## 与 watcher 脚本方案的对比结论

watcher 方案可以作为兜底备选，但不作为推荐实现。

推荐主方案仍然是 wrapper 注入 callback，理由如下：

- 更贴近 datatunerx 现有实现思路
- 指标源直接来自 TrainerState，不依赖文件追尾
- 多卡去重更自然
- 不需要双进程编排
- 只改自定义镜像和 Job 入口即可

## 实施顺序建议

1. 先完成自定义镜像 overlay 设计
2. 再实现 wrapper 入口和 callback
3. 再补 remote write 客户端
4. 最后改 [examples/llamafactory/lora-sft-demo.yaml](/Users/jiang/workspace/Projects/erikjiang/trainer/examples/llamafactory/lora-sft-demo.yaml) 并做单卡验证
5. 单卡通过后再验证多卡去重和 remote write 容错

## 范围说明

本次方案包含：

- demo Job 的 loss 和 eval_loss remote write 上报设计
- 自定义镜像 overlay 设计
- wrapper callback 设计
- remote write 客户端设计
- Job YAML 改造点设计

本次方案不包含：

- Pushgateway
- sidecar exporter
- Operator 层自动注入逻辑
- Trainer 内建 runtime 正式产品化
- 新的集群级组件
