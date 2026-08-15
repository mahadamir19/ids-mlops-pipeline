from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.training.baselines import (
    collect_reproducibility_lineage,
    configure_mlflow_runtime_environment,
    dvc_status_is_clean,
    sha256_file,
)


class Phase2LineageTests(unittest.TestCase):
    def test_sha256_file_hashes_binary_content(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase2_lineage"
        path = root / "payload.bin"
        try:
            root.mkdir(parents=True, exist_ok=True)
            payload = b"sentinelml\x00lineage\n"
            path.write_bytes(payload)

            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())
        finally:
            if path.exists():
                path.unlink()
            if root.exists():
                shutil.rmtree(root)

    def test_dvc_status_clean_interpretation(self) -> None:
        self.assertTrue(dvc_status_is_clean(""))
        self.assertTrue(dvc_status_is_clean("Data and pipelines are up to date.\n"))
        self.assertFalse(dvc_status_is_clean("stage changed: phase2_baselines\n"))

    def test_configure_mlflow_runtime_environment_sets_server_defaults(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase2_mlflow_runtime"
        env_path = root / "docker" / ".env"
        try:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text(
                "\n".join(
                    [
                        "MINIO_ROOT_USER=sentinelml",
                        "MINIO_ROOT_PASSWORD=minio123",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                tracking_uri = configure_mlflow_runtime_environment(repo_root=root)

                self.assertEqual(tracking_uri, "http://127.0.0.1:5000")
                self.assertEqual(
                    os.environ["MLFLOW_TRACKING_URI"],
                    "http://127.0.0.1:5000",
                )
                self.assertEqual(
                    os.environ["MLFLOW_S3_ENDPOINT_URL"],
                    "http://127.0.0.1:9000",
                )
                self.assertEqual(os.environ["AWS_ACCESS_KEY_ID"], "sentinelml")
                self.assertEqual(os.environ["AWS_SECRET_ACCESS_KEY"], "minio123")
                self.assertEqual(os.environ["AWS_DEFAULT_REGION"], "us-east-1")
                self.assertEqual(os.environ["TMP"], str(root / ".tmp" / "mlflow-temp"))
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_collect_lineage_uses_repository_commands_and_hashes(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase2_lineage_collect"
        dvc_lock = root / "dvc.lock"
        training_config = root / "training_config.yaml"
        feature_schema = root / "feature_schema.json"
        try:
            root.mkdir(parents=True, exist_ok=True)
            dvc_lock.write_bytes(b"dvc-lock-content")
            training_config.write_bytes(b"random_seed: 42\n")
            feature_schema.write_bytes(b'{"feature_columns":[]}\n')

            outputs = {
                ("git", "rev-parse", "HEAD"): "abc123\n",
                ("git", "rev-parse", "--abbrev-ref", "HEAD"): "main\n",
                ("git", "status", "--porcelain=v1"): " M src/file.py\n",
                ("dvc", "status"): "Data and pipelines are up to date.\n",
            }

            def fake_run(
                args: list[str],
                cwd: Path,
                capture_output: bool,
                text: bool,
                check: bool,
                env: dict[str, str],
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(cwd, root)
                self.assertTrue(capture_output)
                self.assertTrue(text)
                self.assertTrue(check)
                if args[0] == "dvc":
                    self.assertEqual(
                        env["DVC_SITE_CACHE_DIR"],
                        str(root / ".tmp" / "dvc-site-cache"),
                    )
                return subprocess.CompletedProcess(args, 0, outputs[tuple(args)], "")

            with patch("sentinelml.tracking.mlflow.subprocess.run", fake_run):
                lineage = collect_reproducibility_lineage(
                    mlflow_parent_run_id="parent-run-id",
                    training_config_path=training_config,
                    feature_schema_path=feature_schema,
                    repo_root=root,
                )

            self.assertEqual(lineage["mlflow_parent_run_id"], "parent-run-id")
            self.assertEqual(lineage["git_commit"], "abc123")
            self.assertEqual(lineage["git_branch"], "main")
            self.assertTrue(lineage["git_dirty"])
            self.assertEqual(lineage["git_status_porcelain"], " M src/file.py\n")
            self.assertTrue(lineage["dvc_status_clean"])
            self.assertEqual(
                lineage["dvc_lock_sha256"],
                hashlib.sha256(b"dvc-lock-content").hexdigest(),
            )
            self.assertEqual(
                lineage["training_config_sha256"],
                hashlib.sha256(b"random_seed: 42\n").hexdigest(),
            )
            self.assertEqual(
                lineage["feature_schema_sha256"],
                hashlib.sha256(b'{"feature_columns":[]}\n').hexdigest(),
            )
        finally:
            for path in [feature_schema, training_config, dvc_lock]:
                if path.exists():
                    path.unlink()
            if root.exists():
                shutil.rmtree(root)

    def test_collect_lineage_can_tolerate_missing_repository_tools(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase2_lineage_missing_tools"
        training_config = root / "training_config.yaml"
        feature_schema = root / "feature_schema.json"
        try:
            root.mkdir(parents=True, exist_ok=True)
            training_config.write_bytes(b"random_seed: 42\n")
            feature_schema.write_bytes(b'{"feature_columns":[]}\n')

            def missing_run(*args: object, **kwargs: object) -> None:
                raise FileNotFoundError("git")

            with patch("sentinelml.tracking.mlflow.subprocess.run", missing_run):
                lineage = collect_reproducibility_lineage(
                    mlflow_parent_run_id="parent-run-id",
                    training_config_path=training_config,
                    feature_schema_path=feature_schema,
                    repo_root=root,
                    strict_repository_tools=False,
                )

            self.assertIsNone(lineage["git_commit"])
            self.assertIsNone(lineage["git_branch"])
            self.assertIsNone(lineage["git_dirty"])
            self.assertIsNone(lineage["dvc_status_clean"])
            self.assertIsNone(lineage["dvc_lock_sha256"])
            self.assertIn("git rev-parse HEAD", lineage["repository_command_errors"])
            self.assertEqual(
                lineage["training_config_sha256"],
                hashlib.sha256(b"random_seed: 42\n").hexdigest(),
            )
        finally:
            for path in [feature_schema, training_config]:
                if path.exists():
                    path.unlink()
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
