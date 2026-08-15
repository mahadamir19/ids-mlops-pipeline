"""Validation for labelled production observations before approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinelml.serving.validation import (
    FeatureSchema,
    FeatureValidationError,
    validate_feature_record,
)


@dataclass(frozen=True)
class ProductionValidationResult:
    status: str
    errors: list[dict[str, Any]]

    @property
    def approved(self) -> bool:
        return self.status == "approved"


def validate_prediction_for_production(
    *,
    prediction: dict[str, Any],
    ground_truth: int,
    schema: FeatureSchema,
) -> ProductionValidationResult:
    errors: list[dict[str, Any]] = []

    if not prediction.get("prediction_id"):
        errors.append({"code": "missing_prediction_id"})

    if ground_truth not in {0, 1}:
        errors.append({"code": "invalid_ground_truth", "value": ground_truth})

    features = prediction.get("features")
    if not isinstance(features, dict):
        errors.append({"code": "missing_feature_payload"})
    else:
        try:
            validate_feature_record(features, schema)
        except FeatureValidationError as exc:
            errors.append(
                {
                    "code": "feature_schema_validation_failed",
                    "error": exc.to_error_payload(),
                }
            )

    if not prediction.get("model_version"):
        errors.append({"code": "missing_source_model_version"})

    status = "approved" if not errors else "rejected"
    return ProductionValidationResult(status=status, errors=errors)
