from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.data.preprocess import generate_phase1_datasets


def header() -> list[str]:
    names = [f" Feature {idx}" for idx in range(79)]
    names[0] = " Destination Port"
    names[14] = "Flow Bytes/s"
    names[15] = " Flow Packets/s"
    names[31] = " Bwd PSH Flags"
    names[33] = " Bwd URG Flags"
    names[34] = " Fwd Header Length"
    names[55] = " Fwd Header Length"
    names[56] = "Fwd Avg Bytes/Bulk"
    names[57] = " Fwd Avg Packets/Bulk"
    names[58] = " Fwd Avg Bulk Rate"
    names[59] = " Bwd Avg Bytes/Bulk"
    names[60] = " Bwd Avg Packets/Bulk"
    names[61] = "Bwd Avg Bulk Rate"
    names[78] = " Label"
    return names


CONSTANT_INDICES = [31, 33, 56, 57, 58, 59, 60, 61]


def row(seed: int, label: str, *, inf: bool = False) -> list[str]:
    values = [str(seed + idx + 1) for idx in range(79)]
    for idx in CONSTANT_INDICES:
        values[idx] = "0"
    values[34] = str(seed + 340)
    values[55] = values[34]
    values[14] = "Infinity" if inf else str(seed + 140)
    values[15] = str(seed + 150)
    values[78] = label
    return values


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        ",".join(header()) + "\n" + "\n".join(",".join(item) for item in rows) + "\n",
        encoding="utf-8",
    )


def eda_summary(raw_files: list[Path]) -> dict:
    return {
        "files": [
            {
                "source_filename": path.name,
                "day": path.name.split("-", 1)[0].capitalize(),
            }
            for path in raw_files
        ],
        "original_columns": [
            {"index": idx, "name": name} for idx, name in enumerate(header())
        ],
        "duplicate_original_column_names": [" Fwd Header Length"],
        "duplicate_trimmed_column_names": ["Fwd Header Length"],
        "overall": {
            "constant_features": [
                {"index": idx, "name": header()[idx], "value": "0"}
                for idx in CONSTANT_INDICES
            ],
            "near_constant_features": [{"index": 32, "name": " Fwd URG Flags"}],
        },
    }


class Phase1PreprocessingTests(unittest.TestCase):
    def test_generates_deterministic_clean_temporal_partitions(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase1_fixture"
        if root.exists():
            shutil.rmtree(root)
        try:
            root.mkdir(parents=True)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            monday_duplicate = row(1, "BENIGN")
            files = {
                "Monday-WorkingHours.pcap_ISCX.csv": [
                    monday_duplicate,
                    monday_duplicate,
                    row(10, "BENIGN"),
                ],
                "Tuesday-WorkingHours.pcap_ISCX.csv": [
                    row(20, "FTP-Patator"),
                    row(21, "BENIGN", inf=True),
                ],
                "Wednesday-workingHours.pcap_ISCX.csv": [row(30, "DoS Hulk")],
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": [
                    row(40, "Web Attack \ufffd XSS")
                ],
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": [
                    row(50, "Infiltration")
                ],
                "Friday-WorkingHours-Morning.pcap_ISCX.csv": [monday_duplicate],
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": [row(60, "DDoS")],
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": [
                    row(70, "PortScan")
                ],
            }
            raw_paths = []
            before = {}
            for filename, rows in files.items():
                path = raw_dir / filename
                write_csv(path, rows)
                raw_paths.append(path)
                before[filename] = path.read_bytes()

            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(eda_summary(raw_paths)), encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_data_dir": str(raw_dir),
                        "eda_summary_path": str(summary_path),
                        "processed_dir": str(root / "processed"),
                        "reference_dir": str(root / "reference"),
                        "report_dir": str(root / "reports"),
                        "output_format": "parquet",
                        "chunk_rows": 2,
                        "reference_fraction": 1.0,
                        "reference_hash_modulus": 1000000,
                        "split_strategy": {
                            "name": "temporal_weekday_holdout",
                            "assignment": {
                                "train": ["Monday", "Tuesday", "Wednesday"],
                                "validation": ["Thursday"],
                                "test": ["Friday"],
                            },
                        },
                        "duplicate_column_policy": {
                            "trimmed_name": "Fwd Header Length",
                            "keep_original_index": 34,
                            "drop_original_index": 55,
                            "require_value_equivalence_before_drop": True,
                        },
                        "constant_feature_policy": {"action": "drop"},
                        "near_constant_feature_policy": {
                            "action": "retain_for_initial_baseline"
                        },
                        "destination_port_policy": {
                            "action": "retain_for_initial_baseline",
                            "warning": "review leakage risk",
                        },
                        "invalid_numeric_policy": {"missing_values": "drop_rows"},
                        "provenance_columns": [
                            "source_file",
                            "source_day",
                            "source_partition",
                            "source_row_number",
                            "row_digest",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            mapping_path = root / "mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "known_original_labels": [
                            "BENIGN",
                            "FTP-Patator",
                            "DoS Hulk",
                            "Web Attack \ufffd XSS",
                            "Infiltration",
                            "DDoS",
                            "PortScan",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            first_manifest = generate_phase1_datasets(
                config_path=config_path, label_mapping_path=mapping_path
            )
            first_train = pd.read_parquet(root / "processed" / "train.parquet")
            second_manifest = generate_phase1_datasets(
                config_path=config_path, label_mapping_path=mapping_path
            )
            second_train = pd.read_parquet(root / "processed" / "train.parquet")

            for filename, content in before.items():
                self.assertEqual(content, (raw_dir / filename).read_bytes())

            self.assertEqual(first_manifest["partitions"], second_manifest["partitions"])
            pd.testing.assert_frame_equal(first_train, second_train)

            train = pd.read_parquet(root / "processed" / "train.parquet")
            validation = pd.read_parquet(root / "processed" / "validation.parquet")
            test = pd.read_parquet(root / "processed" / "test.parquet")
            reference = pd.read_parquet(root / "reference" / "reference.parquet")
            schema = json.loads((root / "reports" / "feature_schema.json").read_text())
            features = schema["feature_columns"]

            self.assertEqual(set(train["source_day"]), {"Monday", "Tuesday", "Wednesday"})
            self.assertEqual(set(validation["source_day"]), {"Thursday"})
            self.assertEqual(set(test["source_day"]), {"Friday"})
            self.assertTrue(set(reference["row_digest"]).issubset(set(train["row_digest"])))
            self.assertFalse(set(validation["row_digest"]) & set(test["row_digest"]))
            self.assertEqual(list(train[features].columns), list(validation[features].columns))
            self.assertEqual(list(train[features].columns), list(test[features].columns))
            self.assertFalse(train[features].isna().any().any())
            self.assertFalse(np.isinf(train[features].to_numpy(dtype=float)).any())
            self.assertEqual(set(train["target"]).union(validation["target"], test["target"]), {0, 1})
            self.assertFalse({"target", "original_label", "source_file"} & set(features))
            for removed in [
                "bwd_psh_flags",
                "bwd_urg_flags",
                "fwd_avg_bytes_bulk",
                "fwd_avg_packets_bulk",
                "fwd_avg_bulk_rate",
                "bwd_avg_bytes_bulk",
                "bwd_avg_packets_bulk",
                "bwd_avg_bulk_rate",
                "fwd_header_length_pos_55",
            ]:
                self.assertNotIn(removed, features)
            self.assertIn("destination_port", features)
            self.assertTrue((root / "reports" / "validation_raw.json").exists())
            self.assertTrue((root / "reports" / "validation_processed.json").exists())
            self.assertEqual(first_manifest["row_removals"]["duplicate_raw_observation"], 2)
            self.assertEqual(
                first_manifest["row_removals"]["unresolved_missing_or_infinite_feature"],
                1,
            )
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
