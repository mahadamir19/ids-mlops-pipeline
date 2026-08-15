"""Delayed label scheduling and delivery for Phase 6."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from sentinelml.simulation.client import ApiError


class GroundTruthClient(Protocol):
    def submit_ground_truth(
        self,
        *,
        prediction_id: str,
        ground_truth: int,
    ) -> dict[str, Any]: ...

    def submit_ground_truth_batch(
        self,
        labels: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass
class PendingLabel:
    prediction_id: str
    ground_truth: int
    due_at: float
    attempts: int = 0
    last_error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "ground_truth": self.ground_truth,
        }


class RandomizedDelayLabeler:
    def __init__(
        self,
        client: GroundTruthClient,
        *,
        seed: int,
        min_delay_seconds: float,
        max_delay_seconds: float,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        self.client = client
        self.rng = random.Random(seed)
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.pending: list[PendingLabel] = []
        self.delivered: list[dict[str, Any]] = []
        self.failed: list[PendingLabel] = []

    def schedule(self, *, prediction_id: str, ground_truth: int, now: float) -> None:
        delay = self.rng.uniform(self.min_delay_seconds, self.max_delay_seconds)
        self.pending.append(
            PendingLabel(
                prediction_id=prediction_id,
                ground_truth=ground_truth,
                due_at=now + delay,
            )
        )

    def deliver_due(self, *, now: float) -> None:
        remaining: list[PendingLabel] = []
        for label in self.pending:
            if label.due_at > now:
                remaining.append(label)
                continue
            if self._deliver_one(label, now=now):
                continue
            if label.attempts >= self.max_attempts:
                self.failed.append(label)
            else:
                remaining.append(label)
        self.pending = remaining

    def drain(self, *, wait: bool) -> None:
        while self.pending:
            now = time.monotonic()
            next_due = min(label.due_at for label in self.pending)
            if wait and next_due > now:
                time.sleep(next_due - now)
            self.deliver_due(now=max(now, next_due))
            if not wait:
                break

    def _deliver_one(self, label: PendingLabel, *, now: float) -> bool:
        label.attempts += 1
        try:
            response = self.client.submit_ground_truth(
                prediction_id=label.prediction_id,
                ground_truth=label.ground_truth,
            )
        except ApiError as exc:
            label.last_error = str(exc)
            if not exc.retryable:
                label.attempts = self.max_attempts
            label.due_at = now + self.retry_delay_seconds
            return False
        self.delivered.append(response)
        return True


class BatchLabeler:
    def __init__(
        self,
        client: GroundTruthClient,
        *,
        batch_size: int,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.pending: list[PendingLabel] = []
        self.delivered: list[dict[str, Any]] = []
        self.failed: list[PendingLabel] = []

    def schedule(self, *, prediction_id: str, ground_truth: int, now: float) -> None:
        self.pending.append(
            PendingLabel(
                prediction_id=prediction_id,
                ground_truth=ground_truth,
                due_at=now,
            )
        )
        if len(self.pending) >= self.batch_size:
            self.deliver_due(now=now)

    def deliver_due(self, *, now: float) -> None:
        ready = [label for label in self.pending if label.due_at <= now]
        if len(ready) < self.batch_size:
            return
        self._deliver_batch(ready[: self.batch_size], now=now)

    def drain(self, *, wait: bool) -> None:
        del wait
        while self.pending:
            size_before = len(self.pending)
            self._deliver_batch(self.pending[: self.batch_size], now=time.monotonic())
            if len(self.pending) == size_before:
                break

    def _deliver_batch(self, labels: list[PendingLabel], *, now: float) -> None:
        for label in labels:
            label.attempts += 1
        try:
            response = self.client.submit_ground_truth_batch(
                [label.to_payload() for label in labels]
            )
        except ApiError as exc:
            for label in labels:
                label.last_error = str(exc)
                if not exc.retryable:
                    label.attempts = self.max_attempts
                label.due_at = now + self.retry_delay_seconds
            exhausted = [
                label for label in labels if label.attempts >= self.max_attempts
            ]
            self.failed.extend(exhausted)
            self.pending = [
                label
                for label in self.pending
                if label not in exhausted
            ]
            return
        self.delivered.append(response)
        delivered_ids = {label.prediction_id for label in labels}
        self.pending = [
            label for label in self.pending if label.prediction_id not in delivered_ids
        ]


class NoopLabeler:
    def __init__(self) -> None:
        self.pending: list[PendingLabel] = []
        self.delivered: list[dict[str, Any]] = []
        self.failed: list[PendingLabel] = []

    def schedule(self, *, prediction_id: str, ground_truth: int, now: float) -> None:
        del prediction_id, ground_truth, now

    def deliver_due(self, *, now: float) -> None:
        del now

    def drain(self, *, wait: bool) -> None:
        del wait
