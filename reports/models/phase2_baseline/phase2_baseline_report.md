# Phase 2 Baseline ML Report

Generated at `2026-08-12T07:42:33.939044+00:00`.
Run mode: `full`.

## Data Usage
- train: 1,526,504 of 1,526,504 rows; target distribution={'0': 1323598, '1': 202906}
- validation: 398,434 of 398,434 rows; target distribution={'0': 396255, '1': 2179}
- test: 595,860 of 595,860 rows; target distribution={'0': 375204, '1': 220656}

## Validation Comparison
| Model | Score | Precision | Attack Recall | F1 | PR-AUC | ROC-AUC | FPR | FNR | Train s | Latency ms/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.889437 | 0.934760 | 0.841670 | 0.885776 | 0.896189 | 0.993770 | 0.000323 | 0.158330 | 288.254769 | 0.009362 |
| hist_gradient_boosting | 0.729724 | 0.585061 | 0.815971 | 0.681487 | 0.498757 | 0.992321 | 0.003182 | 0.184029 | 127.607374 | 0.019814 |
| random_forest | 0.698802 | 0.860627 | 0.566774 | 0.683453 | 0.720473 | 0.994745 | 0.000505 | 0.433226 | 1123.679275 | 0.013252 |
| logistic_regression | 0.146586 | 0.002610 | 0.011932 | 0.004283 | 0.023979 | 0.865978 | 0.025075 | 0.988068 | 91.552065 | 0.000364 |

## Recommended Baseline
- Model: `xgboost`
- Selection score: `0.889437`
- Rationale: Selected from validation behavior using attack recall, F1, PR-AUC, ROC-AUC, and false-positive rate. Accuracy is reported but not used as the ranking criterion.

## Selected Baseline Test Assessment
- rows: `595860`
- accuracy: `0.657952`
- precision: `0.995995`
- recall: `0.076640`
- attack_class_recall: `0.076640`
- f1: `0.142328`
- pr_auc: `0.831774`
- roc_auc: `0.792083`
- false_positive_rate: `0.000181`
- false_negative_rate: `0.923360`
- inference_time_seconds: `6.448422`
- inference_latency_ms_per_row: `0.010822`
- confusion_matrix: `{'tn': 375136, 'fp': 68, 'fn': 203745, 'tp': 16911}`

## Feature Importance
### logistic_regression
- #1 `destination_port`: 153.677750
- #2 `packet_length_mean`: 33.002068
- #3 `average_packet_size`: 28.981358
- #4 `flow_duration`: 25.524031
- #5 `active_max`: 24.672594
- #6 `flow_iat_max`: 23.258703
- #7 `fwd_iat_total`: 20.572906
- #8 `fwd_packet_length_std`: 15.320149
- #9 `active_mean`: 13.318398
- #10 `fwd_iat_max`: 13.083343
- destination_port rank: #1 (153.677750)
### random_forest
- #1 `packet_length_variance`: 0.095375
- #2 `avg_bwd_segment_size`: 0.091202
- #3 `bwd_packet_length_std`: 0.082920
- #4 `packet_length_std`: 0.078170
- #5 `destination_port`: 0.058731
- #6 `packet_length_mean`: 0.052738
- #7 `max_packet_length`: 0.048069
- #8 `bwd_packet_length_max`: 0.045531
- #9 `average_packet_size`: 0.033009
- #10 `subflow_bwd_bytes`: 0.029320
- destination_port rank: #5 (0.058731)
### xgboost
- #1 `bwd_packet_length_std`: 0.577758
- #2 `bwd_packet_length_mean`: 0.087597
- #3 `packet_length_std`: 0.054736
- #4 `destination_port`: 0.054650
- #5 `avg_bwd_segment_size`: 0.048714
- #6 `packet_length_variance`: 0.029152
- #7 `flow_iat_std`: 0.011928
- #8 `fwd_iat_max`: 0.011057
- #9 `idle_mean`: 0.010990
- #10 `min_packet_length`: 0.009560
- destination_port rank: #4 (0.054650)
### hist_gradient_boosting
- No native global feature-importance mechanism was used for this model.

## Scope Notes
- No promotion thresholds were defined.
- No DVC stages, MLflow, Optuna, model registry, serving, monitoring, or CI infrastructure were added.
- `destination_port` was retained and inspected, not removed.
