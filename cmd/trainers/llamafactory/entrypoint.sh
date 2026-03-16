#!/bin/sh
set -e

# Bridge Kubeflow PET_ environment variables to LLaMA Factory expected variable names.
# Kubeflow Torch Plugin injects PET_* prefixed variables (PyTorch Elastic convention),
# but LLaMA Factory launcher reads unprefixed variables (NNODES, MASTER_ADDR, etc.).
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export NPROC_PER_NODE=${PET_NPROC_PER_NODE:-1}
export MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${PET_MASTER_PORT:-29400}
export FORCE_TORCHRUN=1

echo "[entrypoint] Distributed config: NNODES=$NNODES, NODE_RANK=$NODE_RANK, NPROC_PER_NODE=$NPROC_PER_NODE, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

exec "$@"
