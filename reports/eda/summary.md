# CICIDS2017 Phase 1 EDA Summary

Generated from `data/raw/cicids2017` at `2026-08-11T17:31:14.719439+00:00` with streaming CSV scans. Raw files were not modified.

## Dataset Inventory
| File | MiB | Rows | BENIGN | ATTACK | Duplicate rows |
| --- | --- | --- | --- | --- | --- |
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 73.551 | 225,745 | 97,718 | 128,027 | 2,633 |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 73.343 | 286,467 | 127,537 | 158,930 | 72,353 |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | 55.615 | 191,033 | 189,067 | 1,966 | 6,888 |
| Monday-WorkingHours.pcap_ISCX.csv | 168.732 | 529,918 | 529,918 | 0 | 26,935 |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 79.253 | 288,602 | 288,566 | 36 | 35,630 |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 49.613 | 170,366 | 168,186 | 2,180 | 6,066 |
| Tuesday-WorkingHours.pcap_ISCX.csv | 128.821 | 445,909 | 432,074 | 13,835 | 24,065 |
| Wednesday-workingHours.pcap_ISCX.csv | 214.735 | 692,703 | 440,031 | 252,672 | 81,909 |

Overall: 8 CSV files, 2,830,743 rows, 843.664 MiB raw CSV, and 308,381 duplicate rows by exact raw-line digest.

## Columns and Schema
All files match the reference header exactly: `True`.
Column count: 79. Duplicate original column names: `[' Fwd Header Length']`. Duplicate trimmed names: `['Fwd Header Length']`.
Columns with leading/trailing whitespace: 65. Exact names and whitespace metadata are in `summary.json`.

## Labels
| Label | Rows | Share |
| --- | --- | --- |
| BENIGN | 2,273,097 | 80.300% |
| Bot | 1,966 | 0.069% |
| DDoS | 128,027 | 4.523% |
| DoS GoldenEye | 10,293 | 0.364% |
| DoS Hulk | 231,073 | 8.163% |
| DoS Slowhttptest | 5,499 | 0.194% |
| DoS slowloris | 5,796 | 0.205% |
| FTP-Patator | 7,938 | 0.280% |
| Heartbleed | 11 | 0.000% |
| Infiltration | 36 | 0.001% |
| PortScan | 158,930 | 5.614% |
| SSH-Patator | 5,897 | 0.208% |
| Web Attack � Brute Force | 1,507 | 0.053% |
| Web Attack � Sql Injection | 21 | 0.001% |
| Web Attack � XSS | 652 | 0.023% |

## Labels by File
| File | Labels present |
| --- | --- |
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | BENIGN, DDoS |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | BENIGN, PortScan |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | BENIGN, Bot |
| Monday-WorkingHours.pcap_ISCX.csv | BENIGN |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | BENIGN, Infiltration |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | BENIGN, Web Attack � Brute Force, Web Attack � Sql Injection, Web Attack � XSS |
| Tuesday-WorkingHours.pcap_ISCX.csv | BENIGN, FTP-Patator, SSH-Patator |
| Wednesday-workingHours.pcap_ISCX.csv | BENIGN, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, Heartbleed |

## Binary Distribution
BENIGN: 2,273,097 (80.300%). ATTACK: 557,646 (19.700%). This is an EDA grouping only; the production binary mapping is not finalized here.

## Missing and Infinite Values
| Index | Column | Missing count |
| --- | --- | --- |
| 14 | Flow Bytes/s | 1,358 |

| Index | Column | +inf | -inf |
| --- | --- | --- | --- |
| 14 | Flow Bytes/s | 1,509 | 0 |
| 15 |  Flow Packets/s | 2,867 | 0 |

## Constant and Near-Constant Features
| Index | Column | Value |
| --- | --- | --- |
| 31 |  Bwd PSH Flags | 0 |
| 33 |  Bwd URG Flags | 0 |
| 56 | Fwd Avg Bytes/Bulk | 0 |
| 57 |  Fwd Avg Packets/Bulk | 0 |
| 58 |  Fwd Avg Bulk Rate | 0 |
| 59 |  Bwd Avg Bytes/Bulk | 0 |
| 60 |  Bwd Avg Packets/Bulk | 0 |
| 61 | Bwd Avg Bulk Rate | 0 |

| Index | Column | Dominant value | Dominant share |
| --- | --- | --- | --- |
| 32 |  Fwd URG Flags | 0 | 99.9889% |
| 45 |  RST Flag Count | 0 | 99.9758% |
| 49 |  CWE Flag Count | 0 | 99.9889% |
| 50 |  ECE Flag Count | 0 | 99.9757% |

## Suspicious Identifier or Leakage-Prone Columns
| Index | Column | Reason |
| --- | --- | --- |
| 0 | Destination Port | network endpoint/service attribute can encode scenario identity |
| 31 | Bwd PSH Flags | constant across all discovered raw files |
| 32 | Fwd URG Flags | near-constant at 99.9889% |
| 33 | Bwd URG Flags | constant across all discovered raw files |
| 34 | Fwd Header Length | duplicate column name appears at multiple positions |
| 45 | RST Flag Count | near-constant at 99.9758% |
| 49 | CWE Flag Count | near-constant at 99.9889% |
| 50 | ECE Flag Count | near-constant at 99.9757% |
| 55 | Fwd Header Length | duplicate column name appears at multiple positions |
| 56 | Fwd Avg Bytes/Bulk | constant across all discovered raw files |
| 57 | Fwd Avg Packets/Bulk | constant across all discovered raw files |
| 58 | Fwd Avg Bulk Rate | constant across all discovered raw files |
| 59 | Bwd Avg Bytes/Bulk | constant across all discovered raw files |
| 60 | Bwd Avg Packets/Bulk | constant across all discovered raw files |
| 61 | Bwd Avg Bulk Rate | constant across all discovered raw files |
| 78 | Label | target-like name; keep out of feature matrix |

## Memory and Local Resource Notes
Raw CSV size is 843.664 MiB. A dense float64 feature matrix excluding the label would be about 1684.555 MiB before dataframe/index/intermediate overhead, so naive full concatenation can require multiple GB of RAM.

## Recommended Columns to Remove or Review
| Column | Reason |
| --- | --- |
|  Bwd PSH Flags | constant across all discovered raw files; no predictive variance |
|  Bwd URG Flags | constant across all discovered raw files; no predictive variance |
| Fwd Avg Bytes/Bulk | constant across all discovered raw files; no predictive variance |
|  Fwd Avg Packets/Bulk | constant across all discovered raw files; no predictive variance |
|  Fwd Avg Bulk Rate | constant across all discovered raw files; no predictive variance |
|  Bwd Avg Bytes/Bulk | constant across all discovered raw files; no predictive variance |
|  Bwd Avg Packets/Bulk | constant across all discovered raw files; no predictive variance |
| Bwd Avg Bulk Rate | constant across all discovered raw files; no predictive variance |
|  Fwd URG Flags | near-constant (99.9889% dominant value); review before retention |
|  RST Flag Count | near-constant (99.9758% dominant value); review before retention |
|  CWE Flag Count | near-constant (99.9889% dominant value); review before retention |
|  ECE Flag Count | near-constant (99.9757% dominant value); review before retention |
| one duplicate Fwd Header Length column | duplicate feature name appears at multiple positions; resolve by position |
| Label | target column; retain as target, exclude from feature matrix |
| Destination Port | not automatic removal, but service/scenario identity may be leakage-prone |

## Explicit Cleaning Operations Required
- Trim or canonicalize column names in processed data while preserving a raw-name mapping.
- Resolve the duplicate `Fwd Header Length` columns by explicit position.
- Convert features to numeric types only after reporting all original labels.
- Replace or reject positive infinite values in `Flow Bytes/s` and `Flow Packets/s` by policy.
- Handle missing/NaN values in `Flow Bytes/s` before training.
- Drop or quarantine exact duplicate rows according to validation policy, not during raw EDA.
- Create binary `BENIGN`/`ATTACK` target only after preserving multiclass label inventory.

## Possible Day/File-Based Split Strategies
### Temporal weekday holdout
Train: Monday + Tuesday + Wednesday. Validation: Thursday files. Test: Friday files.
Label coverage: Train covers BENIGN, FTP-Patator, SSH-Patator, DoS family, Heartbleed; validation covers Web Attack/Infiltration; test covers Bot/DDoS/PortScan.
Trade-off: Best temporal isolation, but several attack families are unseen during training.

### Attack-family-aware file holdout
Train: Monday + Tuesday + Wednesday + Thursday morning + Friday DDoS. Validation: Thursday Infiltration + Friday Bot. Test: Friday PortScan.
Label coverage: Train includes most high-volume attacks; validation exercises rare Bot/Infiltration; test isolates PortScan.
Trade-off: Better train-label coverage, weaker temporal purity because Friday is split by file.

### Rare-attack stress holdout
Train: Monday + Tuesday + Wednesday + Friday DDoS/PortScan. Validation: Friday Bot. Test: Thursday Web Attack/Infiltration.
Label coverage: Common attacks are learned; rare web/infiltration labels are held out as stress cases.
Trade-off: Useful stress test, but non-chronological and rare-label metrics may be unstable.

## Recommended Split Proposal Requiring Review
Proposed default for the next Phase 1 split implementation: **Temporal weekday holdout**. It best follows the spec preference for temporal/file-based isolation and avoids naive random row-level leakage. The review decision is whether the first baseline should accept stricter unseen-attack-family evaluation or temporarily prioritize broader train-label coverage.
