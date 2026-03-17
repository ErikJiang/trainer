import time
from logging import Logger
from typing import Dict, Optional

import requests
import snappy

from .prometheus_pb2 import TimeSeries, WriteRequest


class PrometheusRemoteWriter:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0, logger: Optional[Logger] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.logger = logger

    def write_metrics(self, metrics: Dict[str, float], labels: Dict[str, str], timestamp_ms: Optional[int] = None) -> None:
        if not metrics:
            return

        timestamp_ms = timestamp_ms or int(time.time() * 1000)
        series = [self._build_timeseries(name, value, labels, timestamp_ms) for name, value in metrics.items()]

        request = WriteRequest()
        request.timeseries.extend(series)
        payload = snappy.compress(request.SerializeToString())

        try:
            response = requests.post(
                self._get_write_url(),
                headers={
                    "Content-Encoding": "snappy",
                    "Content-Type": "application/x-protobuf",
                    "X-Prometheus-Remote-Write-Version": "0.1.0",
                    "User-Agent": "llamafactory-prometheus-remote-write",
                },
                data=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as error:  # noqa: BLE001
            if self.logger is not None:
                self.logger.warning("Prometheus remote write failed: %s", error)

    def _get_write_url(self) -> str:
        if self.base_url.endswith("/api/v1/write"):
            return self.base_url

        return f"{self.base_url}/api/v1/write"

    @staticmethod
    def _build_timeseries(metric_name: str, metric_value: float, labels: Dict[str, str], timestamp_ms: int) -> TimeSeries:
        series = TimeSeries()

        name_label = series.labels.add()
        name_label.name = "__name__"
        name_label.value = metric_name

        for key, value in sorted(labels.items()):
            if value is None or value == "":
                continue

            label = series.labels.add()
            label.name = key
            label.value = str(value)

        sample = series.samples.add()
        sample.value = float(metric_value)
        sample.timestamp = int(timestamp_ms)
        return series
