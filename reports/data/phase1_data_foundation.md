# Phase 1 Data Foundation Report

Generated at `2026-08-12T06:35:11.101126+00:00`.

## Outputs
- `train`: `C:\MLOps-Project\project\data\processed\train.parquet`
- `validation`: `C:\MLOps-Project\project\data\processed\validation.parquet`
- `test`: `C:\MLOps-Project\project\data\processed\test.parquet`
- `reference`: `C:\MLOps-Project\project\data\reference\reference.parquet`

## Row Counts and Binary Distributions
- train: 1,526,504 rows; BENIGN=1,323,598, ATTACK=202,906
- validation: 398,434 rows; BENIGN=396,255, ATTACK=2,179
- test: 595,860 rows; BENIGN=375,204, ATTACK=220,656
- reference: 152,695 rows; BENIGN=132,459, ATTACK=20,236

## Preprocessing Decisions
- Trimmed/canonicalized raw column names and stored original-to-canonical mapping.
- Verified duplicated Fwd Header Length columns are equivalent before dropping original index 55.
- Dropped globally constant features identified by Phase 1 EDA.
- Retained near-constant features for initial baseline.
- Retained Destination Port with documented leakage/scenario-identity warning.
- Dropped rows with unresolved missing or positive/negative infinite feature values.
- Removed exact duplicate raw observations globally before partition assignment.
- Mapped BENIGN to 0 and every known non-BENIGN label to 1.
- Preserved source-file/day provenance columns while excluding them from feature schema.

## Removed Features
- `bwd_psh_flags` from raw index 31 (` Bwd PSH Flags`): globally constant feature identified by Phase 1 EDA
- `bwd_urg_flags` from raw index 33 (` Bwd URG Flags`): globally constant feature identified by Phase 1 EDA
- `fwd_header_length_pos_55` from raw index 55 (` Fwd Header Length`): duplicate Fwd Header Length column dropped after value-equivalence check
- `fwd_avg_bytes_per_bulk` from raw index 56 (`Fwd Avg Bytes/Bulk`): globally constant feature identified by Phase 1 EDA
- `fwd_avg_packets_per_bulk` from raw index 57 (` Fwd Avg Packets/Bulk`): globally constant feature identified by Phase 1 EDA
- `fwd_avg_bulk_rate` from raw index 58 (` Fwd Avg Bulk Rate`): globally constant feature identified by Phase 1 EDA
- `bwd_avg_bytes_per_bulk` from raw index 59 (` Bwd Avg Bytes/Bulk`): globally constant feature identified by Phase 1 EDA
- `bwd_avg_packets_per_bulk` from raw index 60 (` Bwd Avg Packets/Bulk`): globally constant feature identified by Phase 1 EDA
- `bwd_avg_bulk_rate` from raw index 61 (`Bwd Avg Bulk Rate`): globally constant feature identified by Phase 1 EDA

## Retained Review Features
- `destination_port`: Destination Port can encode service or scenario identity and should be reviewed in later leakage experiments.
- `Fwd URG Flags`: near-constant feature retained for initial baseline
- `RST Flag Count`: near-constant feature retained for initial baseline
- `CWE Flag Count`: near-constant feature retained for initial baseline
- `ECE Flag Count`: near-constant feature retained for initial baseline

## Final Feature List
- `destination_port`
- `flow_duration`
- `total_fwd_packets`
- `total_backward_packets`
- `total_length_of_fwd_packets`
- `total_length_of_bwd_packets`
- `fwd_packet_length_max`
- `fwd_packet_length_min`
- `fwd_packet_length_mean`
- `fwd_packet_length_std`
- `bwd_packet_length_max`
- `bwd_packet_length_min`
- `bwd_packet_length_mean`
- `bwd_packet_length_std`
- `flow_bytes_per_s`
- `flow_packets_per_s`
- `flow_iat_mean`
- `flow_iat_std`
- `flow_iat_max`
- `flow_iat_min`
- `fwd_iat_total`
- `fwd_iat_mean`
- `fwd_iat_std`
- `fwd_iat_max`
- `fwd_iat_min`
- `bwd_iat_total`
- `bwd_iat_mean`
- `bwd_iat_std`
- `bwd_iat_max`
- `bwd_iat_min`
- `fwd_psh_flags`
- `fwd_urg_flags`
- `fwd_header_length`
- `bwd_header_length`
- `fwd_packets_per_s`
- `bwd_packets_per_s`
- `min_packet_length`
- `max_packet_length`
- `packet_length_mean`
- `packet_length_std`
- `packet_length_variance`
- `fin_flag_count`
- `syn_flag_count`
- `rst_flag_count`
- `psh_flag_count`
- `ack_flag_count`
- `urg_flag_count`
- `cwe_flag_count`
- `ece_flag_count`
- `down_per_up_ratio`
- `average_packet_size`
- `avg_fwd_segment_size`
- `avg_bwd_segment_size`
- `subflow_fwd_packets`
- `subflow_fwd_bytes`
- `subflow_bwd_packets`
- `subflow_bwd_bytes`
- `init_win_bytes_forward`
- `init_win_bytes_backward`
- `act_data_pkt_fwd`
- `min_seg_size_forward`
- `active_mean`
- `active_std`
- `active_max`
- `active_min`
- `idle_mean`
- `idle_std`
- `idle_max`
- `idle_min`

## Validation Reports
- Raw validation: `C:\MLOps-Project\project\reports\data\validation_raw.json`
- Processed validation: `C:\MLOps-Project\project\reports\data\validation_processed.json`
- Feature schema: `C:\MLOps-Project\project\reports\data\feature_schema.json`

## Warnings
- The approved split intentionally leaves several attack families unseen in training; this is a temporal generalization trade-off, not a random IID benchmark.
- Destination Port remains available for the first baseline but should be tested later as a leakage/scenario-identity ablation.
