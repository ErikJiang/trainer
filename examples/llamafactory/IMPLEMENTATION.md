# LlamaFactory Prometheus Wrapper 实现说明

本文档面向后续维护者，解释 [examples/llamafactory/llamafactory_ext](examples/llamafactory/llamafactory_ext) 中各个文件的职责和调用链。

## 总体思路

实现目标不是改写 LlamaFactory 的训练主流程，而是在现有 torchrun 启动链路上插入一个额外的 Python 模块入口。这个入口负责创建自定义 callback，然后把 callback 注入到 LlamaFactory 的 run_exp 中。

这样做有三个原因：

1. LlamaFactory 本身已经支持外部传入 callbacks。
2. 训练 loss 最自然的采集点就是 TrainerCallback.on_log。
3. 这种方式比旁路 watcher 更稳，也比直接修改上游源代码更容易维护。

## 文件职责

### [examples/llamafactory/llamafactory_ext/train.py](examples/llamafactory/llamafactory_ext/train.py)

这是 torchrun 的新模块入口。

职责只有两件事：

1. 从环境变量中读取 remote write 配置。
2. 构造 PrometheusRemoteWriteCallback，并调用 LlamaFactory 的 run_exp。

它本身不解析训练 YAML，也不参与模型、数据集、分布式初始化逻辑。训练配置仍然由原始的 LlamaFactory 参数体系处理。

### [examples/llamafactory/llamafactory_ext/callbacks.py](examples/llamafactory/llamafactory_ext/callbacks.py)

这里实现了 PrometheusRemoteWriteCallback。

调用链如下：

1. on_train_begin：记录 start_time、max_steps，并创建单线程线程池。
2. on_log：从 state.log_history[-1] 读取最新训练日志。
3. on_log：根据日志内容判断是 train 还是 eval 阶段。
4. on_log：组装低基数 labels 和数值型 metrics。
5. on_log：调用 remote_write.py 中的客户端发送到 Prometheus。
6. on_train_end：关闭线程池。

设计要点：

1. 只在 local process zero 上报，避免多卡重复写入。
2. remote write 走异步单线程池，避免网络波动阻塞训练主线程。
3. 上报失败只记录 warning，不中断训练。

### [examples/llamafactory/llamafactory_ext/remote_write.py](examples/llamafactory/llamafactory_ext/remote_write.py)

这个文件负责实现 Prometheus Remote Write 协议的最小客户端。

发送流程：

1. 构造 TimeSeries。
2. 为每个时间序列写入 __name__ 和业务标签。
3. 把 loss、eval_loss、learning_rate 等数值写入 sample.value。
4. 用 WriteRequest 序列化。
5. 用 snappy 压缩。
6. POST 到 /api/v1/write。

这里和 datatunerx 的一个重要差异是：

不再把 loss 之类的数值塞到 label 中，而是作为 sample.value 上报。这更符合 Prometheus 的数据模型，也避免高基数问题。

### [examples/llamafactory/llamafactory_ext/prometheus_pb2.py](examples/llamafactory/llamafactory_ext/prometheus_pb2.py)

这是 remote write 所需 protobuf 的静态生成文件。

它存在的目的只是为了减少镜像构建依赖，避免在构建阶段引入 protoc。

为了便于维护，目录中同时保留了 [examples/llamafactory/llamafactory_ext/prometheus.proto](examples/llamafactory/llamafactory_ext/prometheus.proto) 作为源文件。后续如果要升级协议定义，应修改 proto 源文件并重新生成 pb2，而不是手工编辑 pb2。

## 调用链

完整调用链如下：

1. Kubernetes Job 执行 torchrun。
2. torchrun 启动模块 llamafactory_ext.train。
3. train.py 调用 run_exp(args=sys.argv[1:], callbacks=[PrometheusRemoteWriteCallback(...)])。
4. run_exp 进入上游 LlamaFactory 训练流程。
5. Hugging Face Trainer 在训练过程中周期性触发 on_log。
6. PrometheusRemoteWriteCallback 从 log_history 中提取指标并 remote write 到 Prometheus。

## 环境变量约定

这个实现只依赖少量环境变量：

- PROMETHEUS_REMOTE_WRITE_URL
- TRAIN_RUN_UID
- PROM_REMOTE_WRITE_JOB_NAME
- PROM_REMOTE_WRITE_NAMESPACE
- PROM_REMOTE_WRITE_TIMEOUT_SECONDS
- PROM_REMOTE_WRITE_MODEL_NAME

如果 PROMETHEUS_REMOTE_WRITE_URL 为空，wrapper 不会注册 callback。这让同一份镜像既可以用于普通训练，也可以用于带 Prometheus 上报的训练。

## 为什么不把代码继续放在 LlamaFactory-repo/src 下

本示例的目标是提供一个自包含 demo，而不是提交一个上游正式模块。Dockerfile、YAML、README、维护文档和 overlay 代码都放在 [examples/llamafactory](examples/llamafactory) 下，有几个好处：

1. 使用者只看一个目录就能找到 Dockerfile、YAML、README 和实现代码。
2. 贡献者不需要先理解整个上游仓库结构。
3. 后续如果这个 demo 要迁移到别的仓库，目录本身就能整体搬走。

## 后续维护建议

1. 如果后续需要新增指标，优先在 callbacks.py 中扩展，不要把业务逻辑塞进 train.py。
2. 如果需要支持更复杂的 Prometheus 标签策略，优先只改 remote_write.py。
3. 如果未来决定将该能力正式上游化，再考虑把 examples 中的 overlay 迁回正式源码目录。
