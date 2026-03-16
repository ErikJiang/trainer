
## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| entrypoint.sh | 新增 | PET_ 环境变量桥接脚本 |
| Dockerfile | 新增 | 基于官方 llamafactory 镜像 + 桥接脚本 |
| build-llamafactory-image.yaml | 新增 | GitHub Actions 构建镜像工作流 |
| llamafactory_lora_sft.yaml | 新增 | ClusterTrainingRuntime 定义 |
| kustomization.yaml | 新增 | Kustomize 注册 |
| lora-sft-configmap.yaml | 新增 | 训练配置 ConfigMap |
| lora-sft-trainjob.yaml | 新增 | TrainJob 提交示例 |
| kustomization.yaml | 修改 | 添加 `llamafactory` 到 resources |

---

## 验证操作说明

### 1. 本地构建镜像验证

```bash
# 在项目根目录执行
docker build . -f cmd/trainers/llamafactory/Dockerfile -t llamafactory-trainer:test

# 验证 entrypoint 桥接脚本正常工作（模拟 Kubeflow 注入的 PET_ 变量）
docker run --rm \
  -e PET_NNODES=2 \
  -e PET_NODE_RANK=0 \
  -e PET_NPROC_PER_NODE=1 \
  -e PET_MASTER_ADDR=10.0.0.1 \
  -e PET_MASTER_PORT=29400 \
  llamafactory-trainer:test \
  sh -c 'echo "NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT FORCE_TORCHRUN=$FORCE_TORCHRUN"'

# 期望输出：
# [entrypoint] Distributed config: NNODES=2, NODE_RANK=0, NPROC_PER_NODE=1, MASTER_ADDR=10.0.0.1, MASTER_PORT=29400
# NNODES=2 NODE_RANK=0 NPROC_PER_NODE=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29400 FORCE_TORCHRUN=1
```

### 2. 验证默认值（不传 PET_ 变量时）

```bash
docker run --rm llamafactory-trainer:test \
  sh -c 'echo "NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"'

# 期望输出：
# [entrypoint] Distributed config: NNODES=1, NODE_RANK=0, NPROC_PER_NODE=1, MASTER_ADDR=127.0.0.1, MASTER_PORT=29400
# NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29400
```

### 3. 验证 llamafactory-cli 可用

```bash
docker run --rm llamafactory-trainer:test llamafactory-cli version

# 期望输出：LLaMA Factory 版本号
```

### 4. GitHub Actions 工作流验证

**自动触发**：推送修改 `cmd/trainers/llamafactory/**` 路径下的文件到任意分支即可触发。对 PR 仅构建不推送，对 master/release 分支推送镜像。

**手动触发**（自定义 tag）：
1. 进入 GitHub 仓库 → Actions → "Build and Push LLaMA Factory Trainer Image"
2. 点击 "Run workflow"
3. 可选填入自定义 `image_tag`（如 `v0.1.0`），留空则使用 git SHA 作为 tag

**镜像发布位置**：
- `ghcr.io/<owner>/trainer/llamafactory-trainer:<tag>`
- `docker.io/<owner>/llamafactory-trainer:<tag>`

### 5. Kustomize 验证

```bash
# 验证 kustomize 可以正常渲染（需安装 kustomize 或 kubectl）
kubectl kustomize manifests/base/runtimes/llamafactory/
# 或
kustomize build manifests/base/runtimes/llamafactory/

# 期望输出：ClusterTrainingRuntime YAML 内容
```

### 6. 集群端到端验证（需要 Kubeflow Trainer 集群 + GPU 节点）

```bash
# 1. 推送镜像到可访问的 registry（或使用 GitHub Actions 构建的镜像）
docker tag llamafactory-trainer:test <your-registry>/llamafactory-trainer:test
docker push <your-registry>/llamafactory-trainer:test

# 2. 更新 ClusterTrainingRuntime 中的镜像地址
#    修改 manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml
#    将 image: ghcr.io/kubeflow/trainer/llamafactory-trainer 改为你的镜像地址

# 3. 部署 Runtime 和 ConfigMap
kubectl apply -f manifests/base/runtimes/llamafactory/llamafactory_lora_sft.yaml
kubectl apply -f examples/llamafactory/lora-sft-configmap.yaml

# 4. 提交 TrainJob
kubectl apply -f examples/llamafactory/lora-sft-trainjob.yaml

# 5. 观察 Pod 启动顺序
kubectl get pods -l batch.kubernetes.io/job-name -w

# 6. 检查桥接变量和分布式初始化
kubectl logs llamafactory-lora-sft-demo-node-0-0 | head -20
# 期望看到 [entrypoint] Distributed config: NNODES=2, NODE_RANK=0, ...

# 7. 检查训练 loss 是否下降
kubectl logs llamafactory-lora-sft-demo-node-0-0 -f | grep "loss"

# 8. 检查 TrainJob 最终状态
kubectl get trainjob llamafactory-lora-sft-demo
# 期望：status 变为 Completed
```

