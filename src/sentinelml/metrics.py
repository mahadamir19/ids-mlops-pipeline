"""Small Prometheus text exposition helpers used by local services."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

LabelValues = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MetricKey:
    name: str
    labels: LabelValues


def normalize_labels(labels: Mapping[str, object] | None = None) -> LabelValues:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


class PrometheusTextRegistry:
    """In-process metrics registry with enough Prometheus exposition for Phase 7."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[MetricKey, float] = defaultdict(float)
        self._gauges: dict[MetricKey, float] = {}
        self._summaries: dict[MetricKey, tuple[int, float]] = {}
        self._help: dict[str, tuple[str, str]] = {}

    def describe(self, name: str, metric_type: str, help_text: str) -> None:
        with self._lock:
            self._help[name] = (metric_type, help_text)

    def inc_counter(
        self,
        name: str,
        amount: float = 1.0,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        if amount < 0:
            raise ValueError("counter increment must be non-negative")
        key = MetricKey(name, normalize_labels(labels))
        with self._lock:
            self._counters[key] += float(amount)

    def set_gauge(
        self,
        name: str,
        value: float | int | bool | None,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        key = MetricKey(name, normalize_labels(labels))
        with self._lock:
            if value is None:
                self._gauges.pop(key, None)
                return
            numeric = float(value)
            if not math.isfinite(numeric):
                self._gauges.pop(key, None)
                return
            self._gauges[key] = numeric

    def clear_gauge_family(self, name: str) -> None:
        with self._lock:
            for key in list(self._gauges):
                if key.name == name:
                    self._gauges.pop(key, None)

    def observe_summary(
        self,
        name: str,
        value: float,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            return
        key = MetricKey(name, normalize_labels(labels))
        with self._lock:
            count, total = self._summaries.get(key, (0, 0.0))
            self._summaries[key] = (count + 1, total + numeric)

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            metric_names = sorted(
                {
                    *[key.name for key in self._counters],
                    *[key.name for key in self._gauges],
                    *[key.name for key in self._summaries],
                }
            )
            for name in metric_names:
                metric_type, help_text = self._help.get(name, ("gauge", name))
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric_type}")
                for key, value in sorted(
                    self._counters.items(), key=lambda item: item[0].labels
                ):
                    if key.name == name:
                        lines.append(format_sample(name, key.labels, value))
                for key, value in sorted(
                    self._gauges.items(), key=lambda item: item[0].labels
                ):
                    if key.name == name:
                        lines.append(format_sample(name, key.labels, value))
                for key, (count, total) in sorted(
                    self._summaries.items(), key=lambda item: item[0].labels
                ):
                    if key.name == name:
                        lines.append(format_sample(f"{name}_count", key.labels, count))
                        lines.append(format_sample(f"{name}_sum", key.labels, total))
            return "\n".join(lines) + "\n"


def format_sample(name: str, labels: LabelValues, value: float | int) -> str:
    label_text = ""
    if labels:
        encoded = ",".join(
            f'{key}="{escape_label_value(value)}"' for key, value in labels
        )
        label_text = f"{{{encoded}}}"
    return f"{name}{label_text} {float(value):.12g}"


def escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
