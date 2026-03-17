import os
import sys
from typing import Optional

from llamafactory.train.tuner import run_exp

from .callbacks import PrometheusRemoteWriteCallback


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def build_callback() -> Optional[PrometheusRemoteWriteCallback]:
    remote_write_url = os.getenv("PROMETHEUS_REMOTE_WRITE_URL", "").strip()
    if not remote_write_url:
        return None

    return PrometheusRemoteWriteCallback(
        remote_write_url=remote_write_url,
        run_uid=os.getenv("TRAIN_RUN_UID", "").strip() or os.getenv("HOSTNAME", "unknown"),
        job_name=os.getenv("PROM_REMOTE_WRITE_JOB_NAME", "llamafactory-lora-sft-demo").strip(),
        namespace=os.getenv("PROM_REMOTE_WRITE_NAMESPACE", "default").strip(),
        model_name=os.getenv("PROM_REMOTE_WRITE_MODEL_NAME", "").strip() or None,
        timeout_seconds=_get_env_float("PROM_REMOTE_WRITE_TIMEOUT_SECONDS", 2.0),
    )


def _run() -> None:
    callback = build_callback()
    callbacks = [callback] if callback is not None else []
    run_exp(args=sys.argv[1:], callbacks=callbacks)


def main() -> None:
    _run()


def _mp_fn(index: int) -> None:
    _run()


if __name__ == "__main__":
    main()
