"""Prediction execution and logging for the loaded champion model."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from sentinelml.serving.database import PredictionRecord
from sentinelml.serving.model_manager import LoadedModel, ModelManager
from sentinelml.serving.repository import PredictionRepository
from sentinelml.training.evaluate import positive_class_scores

BENIGN_LABEL = "BENIGN"
ATTACK_LABEL = "ATTACK"


class PredictionService:
    def __init__(
        self,
        model_manager: ModelManager,
        repository: PredictionRepository,
    ) -> None:
        self.model_manager = model_manager
        self.repository = repository

    def predict_one(self, ordered_features: dict[str, float]) -> dict[str, Any]:
        return self.predict_batch([ordered_features])[0]

    def predict_batch(
        self,
        ordered_records: list[dict[str, float]],
    ) -> list[dict[str, Any]]:
        loaded = self.model_manager.current()
        frame = pd.DataFrame(
            ordered_records,
            columns=loaded.feature_schema.feature_columns,
        )
        start = perf_counter()
        probabilities = attack_probabilities(loaded.model, frame)
        latency_total_ms = (perf_counter() - start) * 1000
        latency_ms = latency_total_ms / len(ordered_records)
        responses: list[dict[str, Any]] = []
        for features, probability in zip(ordered_records, probabilities, strict=True):
            prediction = ATTACK_LABEL if float(probability) >= 0.5 else BENIGN_LABEL
            prediction_id = str(uuid4())
            response = {
                "prediction_id": prediction_id,
                "prediction": prediction,
                "probability": float(probability),
                "model_version": loaded.model_version,
                "latency_ms": float(latency_ms),
            }
            self.repository.persist_prediction(
                build_prediction_record(
                    loaded=loaded,
                    features=features,
                    response=response,
                )
            )
            responses.append(response)
        return responses


def attack_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(positive_class_scores(model, features), dtype=float).reshape(-1)
    if len(scores) != len(features):
        raise ValueError("model probability output length does not match request rows")
    if not np.isfinite(scores).all():
        raise ValueError("model returned non-finite probabilities")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("model returned probabilities outside [0, 1]")
    return scores


def build_prediction_record(
    *,
    loaded: LoadedModel,
    features: dict[str, float],
    response: dict[str, Any],
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=str(response["prediction_id"]),
        timestamp=datetime.now(UTC).isoformat(),
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        model_family=loaded.model_family,
        execution_mode=loaded.execution_mode,
        demo_model=loaded.demo_model,
        features=features,
        prediction=str(response["prediction"]),
        probability=float(response["probability"]),
        latency_ms=float(response["latency_ms"]),
    )
