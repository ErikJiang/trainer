import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from transformers import TrainerCallback

from .remote_write import PrometheusRemoteWriter


logger = logging.getLogger(__name__)

METRIC_NAMES = {
    "loss": "llamafactory_train_loss",
    "eval_loss": "llamafactory_eval_loss",
    "learning_rate": "llamafactory_learning_rate",
    "epoch": "llamafactory_epoch",
}


class PrometheusRemoteWriteCallback(TrainerCallback):
    def __init__(
        self,
        remote_write_url: str,
        run_uid: str,
        job_name: str,
        namespace: str,
        timeout_seconds: float = 2.0,
        model_name: Optional[str] = None,
    ) -> None:
        self.run_uid = run_uid
        self.job_name = job_name
        self.namespace = namespace
        self.model_name = model_name
        self.start_time = 0.0
        self.max_steps = 0
        self.executor: Optional[ThreadPoolExecutor] = None
        self.writer = PrometheusRemoteWriter(remote_write_url, timeout_seconds=timeout_seconds, logger=logger)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._should_report(state):
            return

        self.start_time = time.time()
        self.max_steps = state.max_steps
        self.executor = ThreadPoolExecutor(max_workers=1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def on_log(self, args, state, control, **kwargs):
        if not self._should_report(state):
            return

        if not state.log_history:
            return

        latest_log = state.log_history[-1]
        current_steps = int(state.global_step)
        phase = "eval" if latest_log.get("eval_loss") is not None else "train"
        timestamp_ms = int(time.time() * 1000)
        elapsed_seconds, remaining_seconds = self._get_timing(current_steps)
        base_labels = self._build_labels(phase)
        metrics = self._build_metrics(latest_log, current_steps, elapsed_seconds, remaining_seconds)

        if not metrics:
            return

        if self.executor is not None:
            self.executor.submit(self.writer.write_metrics, metrics, base_labels, timestamp_ms)
        else:
            self.writer.write_metrics(metrics, base_labels, timestamp_ms)

    @staticmethod
    def _append_metric(metrics: Dict[str, float], name: str, value: Any) -> None:
        if value is None:
            return

        try:
            metrics[name] = float(value)
        except (TypeError, ValueError):
            logger.debug("Skip non-numeric metric %s=%r", name, value)

    def _get_progress_percent(self, current_steps: int) -> Optional[float]:
        if self.max_steps <= 0:
            return None

        return round(current_steps / self.max_steps * 100.0, 2)

    def _build_labels(self, phase: str) -> Dict[str, str]:
        labels = {
            "run_uid": self.run_uid,
            "job_name": self.job_name,
            "namespace": self.namespace,
            "phase": phase,
        }
        if self.model_name:
            labels["model_name"] = self.model_name

        return labels

    def _build_metrics(
        self,
        latest_log: Dict[str, Any],
        current_steps: int,
        elapsed_seconds: float,
        remaining_seconds: float,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for log_key, metric_name in METRIC_NAMES.items():
            self._append_metric(metrics, metric_name, latest_log.get(log_key))

        self._append_metric(metrics, "llamafactory_progress_percent", self._get_progress_percent(current_steps))
        self._append_metric(metrics, "llamafactory_current_steps", current_steps)
        self._append_metric(metrics, "llamafactory_total_steps", self.max_steps)
        self._append_metric(metrics, "llamafactory_elapsed_seconds", elapsed_seconds)
        self._append_metric(metrics, "llamafactory_remaining_seconds", remaining_seconds)
        return metrics

    def _get_timing(self, current_steps: int) -> tuple[float, float]:
        elapsed_seconds = max(0.0, time.time() - self.start_time) if self.start_time else 0.0
        remaining_seconds = 0.0
        if current_steps > 0 and self.max_steps > 0:
            avg_seconds_per_step = elapsed_seconds / current_steps
            remaining_seconds = max(0.0, (self.max_steps - current_steps) * avg_seconds_per_step)

        return elapsed_seconds, remaining_seconds

    @staticmethod
    def _should_report(state) -> bool:
        if hasattr(state, "is_local_process_zero") and not state.is_local_process_zero:
            return False

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return local_rank == 0
