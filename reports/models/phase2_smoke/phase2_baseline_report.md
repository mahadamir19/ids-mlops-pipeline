# Phase 2 Baseline ML Report

Generated at `2026-08-13T17:51:52.724192+00:00`.
Run mode: `smoke`.

## Data Usage
- train: 20,000 of 1,526,504 rows; target distribution={'0': 17342, '1': 2658}
- validation: 10,000 of 398,434 rows; target distribution={'0': 9946, '1': 54}
- test: 10,000 of 595,860 rows; target distribution={'0': 6297, '1': 3703}

## Validation Comparison
| Model | Score | Precision | Attack Recall | F1 | PR-AUC | ROC-AUC | FPR | FNR | Train s | Latency ms/row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.265382 | 0.100000 | 0.018519 | 0.031250 | 0.406214 | 0.995798 | 0.000905 | 0.981481 | 0.982717 | 0.003105 |
| hist_gradient_boosting | 0.252715 | 0.230769 | 0.055556 | 0.089552 | 0.247675 | 0.990139 | 0.001005 | 0.944444 | 1.604538 | 0.003140 |
| random_forest | 0.217765 | 0.090909 | 0.037037 | 0.052632 | 0.175734 | 0.978117 | 0.002011 | 0.962963 | 2.007910 | 0.002824 |
| logistic_regression | 0.170663 | 0.008929 | 0.055556 | 0.015385 | 0.031624 | 0.911406 | 0.033481 | 0.944444 | 2.819049 | 0.003180 |

## Recommended Baseline
- Model: `xgboost`
- Selection score: `0.265382`
- Rationale: Selected from validation behavior using attack recall, F1, PR-AUC, ROC-AUC, and false-positive rate. Accuracy is reported but not used as the ranking criterion.

## Selected Baseline Test Assessment
- rows: `10000`
- accuracy: `0.764800`
- precision: `0.998524`
- recall: `0.365379`
- attack_class_recall: `0.365379`
- f1: `0.534994`
- pr_auc: `0.797341`
- roc_auc: `0.804251`
- false_positive_rate: `0.000318`
- false_negative_rate: `0.634621`
- inference_time_seconds: `0.016205`
- inference_latency_ms_per_row: `0.001620`
- confusion_matrix: `{'tn': 6295, 'fp': 2, 'fn': 2350, 'tp': 1353}`

## Feature Importance
### logistic_regression
- #1 `bwd_packet_length_min`: 0.099134
- #2 `min_seg_size_forward`: 0.057161
- #3 `subflow_fwd_packets`: 0.023948
- #4 `act_data_pkt_fwd`: 0.023029
- #5 `total_fwd_packets`: 0.022891
- #6 `min_packet_length`: 0.019886
- #7 `destination_port`: 0.013147
- #8 `fwd_packet_length_min`: 0.010778
- #9 `fin_flag_count`: 0.010443
- #10 `fwd_packet_length_std`: 0.007925
- destination_port rank: #7 (0.013147)
### random_forest
- #1 `packet_length_std`: 0.282786
- #2 `packet_length_variance`: 0.246442
- #3 `bwd_packet_length_std`: 0.195487
- #4 `bwd_packet_length_mean`: 0.053186
- #5 `avg_bwd_segment_size`: 0.038586
- #6 `fwd_iat_mean`: 0.033018
- #7 `destination_port`: 0.027195
- #8 `fwd_iat_min`: 0.014203
- #9 `init_win_bytes_backward`: 0.012706
- #10 `fwd_iat_max`: 0.010852
- destination_port rank: #7 (0.027195)
### xgboost
- #1 `packet_length_variance`: 0.399725
- #2 `packet_length_std`: 0.205139
- #3 `bwd_packet_length_std`: 0.184835
- #4 `bwd_packet_length_mean`: 0.046262
- #5 `destination_port`: 0.034900
- #6 `avg_bwd_segment_size`: 0.022422
- #7 `min_packet_length`: 0.012075
- #8 `flow_iat_std`: 0.009849
- #9 `bwd_packet_length_max`: 0.008557
- #10 `subflow_fwd_bytes`: 0.008386
- destination_port rank: #5 (0.034900)
### hist_gradient_boosting
- #1 `bwd_packet_length_std`: 0.571865
- #2 `bwd_packet_length_mean`: 0.113663
- #3 `destination_port`: 0.107915
- #4 `min_packet_length`: 0.076882
- #5 `init_win_bytes_backward`: 0.019007
- #6 `max_packet_length`: 0.017287
- #7 `fwd_iat_max`: 0.012478
- #8 `urg_flag_count`: 0.012098
- #9 `flow_iat_std`: 0.009915
- #10 `fwd_header_length`: 0.006881
- destination_port rank: #3 (0.107915)

## Scope Notes
- No promotion thresholds were defined.
- No DVC stages, MLflow, Optuna, model registry, serving, monitoring, or CI infrastructure were added.
- `destination_port` was retained and inspected, not removed.
