# Phase 2 Baseline ML Report

Generated at `2026-08-12T07:14:23.479407+00:00`.
Run mode: `smoke`.

## Data Usage
- train: 20,000 of 1,526,504 rows; target distribution={'0': 17342, '1': 2658}
- validation: 10,000 of 398,434 rows; target distribution={'0': 9946, '1': 54}
- test: 10,000 of 595,860 rows; target distribution={'0': 6297, '1': 3703}

## Validation Comparison
| Model | Score | Precision | Attack Recall | F1 | PR-AUC | ROC-AUC | FPR | FNR | Train s | Latency ms/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hist_gradient_boosting | 0.282892 | 0.363636 | 0.074074 | 0.123077 | 0.307696 | 0.993083 | 0.000704 | 0.925926 | 10.781313 | 0.022826 |
| xgboost | 0.281041 | 0.125000 | 0.018519 | 0.032258 | 0.467372 | 0.996872 | 0.000704 | 0.981481 | 5.144954 | 0.006138 |
| random_forest | 0.175091 | 0.000000 | 0.000000 | 0.000000 | 0.114290 | 0.965436 | 0.000503 | 1.000000 | 4.208782 | 0.005483 |
| logistic_regression | 0.145596 | 0.004950 | 0.018519 | 0.007812 | 0.017236 | 0.838632 | 0.020209 | 0.981481 | 0.309244 | 0.000752 |

## Recommended Baseline
- Model: `hist_gradient_boosting`
- Selection score: `0.282892`
- Rationale: Selected from validation behavior using attack recall, F1, PR-AUC, ROC-AUC, and false-positive rate. Accuracy is reported but not used as the ranking criterion.

## Selected Baseline Test Assessment
- rows: `10000`
- accuracy: `0.765000`
- precision: `0.996332`
- recall: `0.366730`
- attack_class_recall: `0.366730`
- f1: `0.536123`
- pr_auc: `0.748631`
- roc_auc: `0.636193`
- false_positive_rate: `0.000794`
- false_negative_rate: `0.633270`
- inference_time_seconds: `0.198114`
- inference_latency_ms_per_row: `0.019811`
- confusion_matrix: `{'tn': 6292, 'fp': 5, 'fn': 2345, 'tp': 1358}`

## Feature Importance
### logistic_regression
- #1 `bwd_packet_length_min`: 8.309752
- #2 `packet_length_variance`: 7.432013
- #3 `destination_port`: 6.822339
- #4 `fwd_packet_length_std`: 4.682166
- #5 `total_length_of_bwd_packets`: 3.546576
- #6 `subflow_bwd_bytes`: 3.544290
- #7 `bwd_header_length`: 3.452155
- #8 `flow_duration`: 2.391461
- #9 `min_seg_size_forward`: 1.998032
- #10 `bwd_iat_total`: 1.855709
- destination_port rank: #3 (6.822339)
### random_forest
- #1 `packet_length_variance`: 0.089264
- #2 `packet_length_std`: 0.083566
- #3 `bwd_packet_length_std`: 0.082905
- #4 `avg_bwd_segment_size`: 0.076631
- #5 `bwd_packet_length_max`: 0.054678
- #6 `packet_length_mean`: 0.054454
- #7 `destination_port`: 0.045605
- #8 `max_packet_length`: 0.044993
- #9 `bwd_packet_length_mean`: 0.042899
- #10 `average_packet_size`: 0.030853
- destination_port rank: #7 (0.045605)
### xgboost
- #1 `bwd_packet_length_std`: 0.385736
- #2 `packet_length_variance`: 0.223463
- #3 `bwd_packet_length_mean`: 0.071080
- #4 `destination_port`: 0.054673
- #5 `avg_bwd_segment_size`: 0.044102
- #6 `min_packet_length`: 0.037317
- #7 `packet_length_std`: 0.015261
- #8 `init_win_bytes_backward`: 0.010285
- #9 `fwd_iat_max`: 0.009947
- #10 `flow_iat_std`: 0.009330
- destination_port rank: #4 (0.054673)
### hist_gradient_boosting
- No native global feature-importance mechanism was used for this model.

## Scope Notes
- No promotion thresholds were defined.
- No DVC stages, MLflow, Optuna, model registry, serving, monitoring, or CI infrastructure were added.
- `destination_port` was retained and inspected, not removed.
