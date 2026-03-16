FROM docker.io/hiyouga/llamafactory:0.9.4

WORKDIR /workspace

# Add PET_ environment variable bridge script for Kubeflow Trainer compatibility.
# Kubeflow Torch Plugin injects PET_* prefixed variables, but LLaMA Factory
# launcher expects unprefixed variables (NNODES, MASTER_ADDR, etc.).
COPY cmd/trainers/llamafactory/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
