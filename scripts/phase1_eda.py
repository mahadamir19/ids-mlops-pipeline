"""Fast, read-only Phase 1 EDA for CICIDS2017 MachineLearningCSV files."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
REPORT_DIR = ROOT / "reports" / "eda"
NOTEBOOK_DIR = ROOT / "notebooks"
SUMMARY_JSON = REPORT_DIR / "summary.json"
SUMMARY_MD = REPORT_DIR / "summary.md"
NOTEBOOK = NOTEBOOK_DIR / "01_eda.ipynb"

NEAR_CONSTANT_THRESHOLD = 0.995
MISSING = {b"", b"na", b"n/a", b"nan", b"null", b"none", b"?"}
POS_INF = {b"inf", b"+inf", b"infinity", b"+infinity"}
NEG_INF = {b"-inf", b"-infinity"}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def pct(part: int, total: int) -> float:
    return part / total * 100.0 if total else 0.0


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def decode(cell: bytes | None) -> str | None:
    if cell is None:
        return None
    return cell.decode("utf-8", errors="replace")


def split_header(raw: bytes) -> list[str]:
    text = raw.decode("utf-8-sig", errors="replace").rstrip("\r\n")
    return text.split(",")


def split_row(raw: bytes, col_count: int) -> list[bytes]:
    row = raw.rstrip(b"\r\n").split(b",")
    if len(row) != col_count:
        row = (row + [b""] * col_count)[:col_count]
    return row


def token_type(cell: bytes) -> str:
    stripped = cell.strip()
    lower = stripped.lower()
    if lower in MISSING:
        return "missing"
    if lower in POS_INF or lower in NEG_INF:
        return "infinite"
    has_digit = False
    has_dot_or_exp = False
    for idx, byte in enumerate(stripped):
        if 48 <= byte <= 57:
            has_digit = True
        elif byte in (43, 45):
            if idx != 0 and stripped[idx - 1] not in (69, 101):
                return "string"
        elif byte == 46:
            has_dot_or_exp = True
        elif byte in (69, 101):
            has_dot_or_exp = True
        else:
            return "string"
    if not has_digit:
        return "string"
    return "float" if has_dot_or_exp else "integer"


def update_majority(state: dict[str, Any], cell: bytes) -> None:
    if state["count"] == 0:
        state["candidate"] = cell
        state["count"] = 1
    elif state["candidate"] == cell:
        state["count"] += 1
    else:
        state["count"] -= 1


def day_and_scenario(filename: str) -> tuple[str, str]:
    stem = filename.replace(".pcap_ISCX.csv", "")
    pieces = stem.split("-")
    day = pieces[0].capitalize()
    return day, stem[len(pieces[0]) + 1 :] if len(pieces) > 1 else "unknown"


def scan_first_pass(paths: list[Path]) -> dict[str, Any]:
    reference: list[str] | None = None
    files = []
    consistency: dict[str, Any] = {
        "reference_file": None,
        "all_files_match_exact_header": True,
        "all_files_match_trimmed_header": True,
        "by_file": {},
    }
    global_seen: set[bytes] = set()
    global_rows = 0
    global_duplicates = 0
    global_labels: Counter[str] = Counter()
    global_binary: Counter[str] = Counter()
    global_dtype: list[Counter[str]] = []
    global_missing: list[int] = []
    global_pos_inf: list[int] = []
    global_neg_inf: list[int] = []
    global_first: list[bytes | None] = []
    global_constant: list[bool] = []
    global_majority: list[dict[str, Any]] = []

    for path in paths:
        started = time.time()
        with path.open("rb") as handle:
            header = split_header(handle.readline())
            col_count = len(header)
            if reference is None:
                reference = header
                consistency["reference_file"] = path.name
                global_dtype = [Counter() for _ in header]
                global_missing = [0 for _ in header]
                global_pos_inf = [0 for _ in header]
                global_neg_inf = [0 for _ in header]
                global_first = [None for _ in header]
                global_constant = [True for _ in header]
                global_majority = [{"candidate": None, "count": 0} for _ in header]
            assert reference is not None
            exact = header == reference
            trimmed = [c.strip() for c in header] == [c.strip() for c in reference]
            consistency["all_files_match_exact_header"] &= exact
            consistency["all_files_match_trimmed_header"] &= trimmed
            consistency["by_file"][path.name] = {
                "column_count": col_count,
                "exact_header_match_reference": exact,
                "trimmed_header_match_reference": trimmed,
                "missing_from_reference_exact": [c for c in reference if c not in header],
                "extra_vs_reference_exact": [c for c in header if c not in reference],
            }
            label_idx = next(i for i, name in enumerate(header) if name.strip().lower() == "label")
            seen: set[bytes] = set()
            duplicates = 0
            row_count = 0
            row_length_mismatches = 0
            labels: Counter[str] = Counter()
            binary: Counter[str] = Counter()
            dtype = [Counter() for _ in header]
            missing = [0 for _ in header]
            pos_inf = [0 for _ in header]
            neg_inf = [0 for _ in header]
            first: list[bytes | None] = [None for _ in header]
            constant = [True for _ in header]
            majority = [{"candidate": None, "count": 0} for _ in header]

            for raw_line in handle:
                line = raw_line.rstrip(b"\r\n")
                row_count += 1
                global_rows += 1
                digest = hashlib.sha256(line).digest()
                if digest in seen:
                    duplicates += 1
                else:
                    seen.add(digest)
                if digest in global_seen:
                    global_duplicates += 1
                else:
                    global_seen.add(digest)

                row = line.split(b",")
                if len(row) != col_count:
                    row_length_mismatches += 1
                    row = (row + [b""] * col_count)[:col_count]

                label = decode(row[label_idx].strip()) or ""
                labels[label] += 1
                global_labels[label] += 1
                class_name = "BENIGN" if label == "BENIGN" else "ATTACK"
                binary[class_name] += 1
                global_binary[class_name] += 1

                for idx, cell in enumerate(row):
                    kind = token_type(cell)
                    dtype[idx][kind] += 1
                    global_dtype[idx][kind] += 1
                    if kind == "missing":
                        missing[idx] += 1
                        global_missing[idx] += 1
                    elif kind == "infinite":
                        if cell.strip().lower() in POS_INF:
                            pos_inf[idx] += 1
                            global_pos_inf[idx] += 1
                        else:
                            neg_inf[idx] += 1
                            global_neg_inf[idx] += 1
                    if first[idx] is None:
                        first[idx] = cell
                    elif constant[idx] and cell != first[idx]:
                        constant[idx] = False
                    if global_first[idx] is None:
                        global_first[idx] = cell
                    elif global_constant[idx] and cell != global_first[idx]:
                        global_constant[idx] = False
                    update_majority(majority[idx], cell)
                    update_majority(global_majority[idx], cell)

        columns = []
        for idx, name in enumerate(header):
            label_col = idx == label_idx
            columns.append(
                {
                    "index": idx,
                    "name": name,
                    "trimmed_name": name.strip(),
                    "dtype_inferred": infer_dtype(dtype[idx], label_col),
                    "dtype_evidence_counts": dict(dtype[idx]),
                    "missing_count": missing[idx],
                    "positive_infinite_count": pos_inf[idx],
                    "negative_infinite_count": neg_inf[idx],
                    "constant": constant[idx],
                    "constant_value": decode(first[idx]) if constant[idx] else None,
                    "majority_candidate": decode(majority[idx]["candidate"]),
                }
            )
        day, scenario = day_and_scenario(path.name)
        files.append(
            {
                "source_filename": path.name,
                "path": rel(path),
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "row_count": row_count,
                "column_count": col_count,
                "day": day,
                "scenario_from_filename": scenario,
                "label_column_index": label_idx,
                "label_column_name": header[label_idx],
                "labels": dict(labels),
                "binary_class_distribution": dict(binary),
                "duplicate_rows": duplicates,
                "row_length_mismatches": row_length_mismatches,
                "columns": columns,
                "scan_seconds_first_pass": round(time.time() - started, 3),
            }
        )
        print(f"first pass: {path.name}: {row_count:,} rows", flush=True)

    assert reference is not None
    original_counts = Counter(reference)
    trimmed_counts = Counter(name.strip() for name in reference)
    total_bytes = sum(f["size_bytes"] for f in files)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_directory": rel(RAW_DIR),
        "scan_method": "byte-level streaming scan; exact duplicate rows counted by SHA-256 digest of each raw data line excluding newline",
        "near_constant_threshold": NEAR_CONSTANT_THRESHOLD,
        "missing_tokens_interpreted_for_detection": sorted(t.decode() for t in MISSING),
        "files": files,
        "original_columns": [
            {
                "index": i,
                "name": name,
                "trimmed_name": name.strip(),
                "leading_whitespace": len(name) - len(name.lstrip()),
                "trailing_whitespace": len(name) - len(name.rstrip()),
            }
            for i, name in enumerate(reference)
        ],
        "duplicate_original_column_names": [
            name for name, count in original_counts.items() if count > 1
        ],
        "duplicate_trimmed_column_names": [
            name for name, count in trimmed_counts.items() if count > 1
        ],
        "columns_with_leading_or_trailing_whitespace": [
            {
                "index": i,
                "original": name,
                "trimmed": name.strip(),
                "leading_whitespace": len(name) - len(name.lstrip()),
                "trailing_whitespace": len(name) - len(name.rstrip()),
            }
            for i, name in enumerate(reference)
            if name != name.strip()
        ],
        "column_consistency": consistency,
        "overall": {
            "file_count": len(files),
            "row_count": global_rows,
            "size_bytes": total_bytes,
            "size_mb": round(total_bytes / (1024 * 1024), 3),
            "labels": dict(global_labels),
            "binary_class_distribution": dict(global_binary),
            "duplicate_rows": global_duplicates,
            "unique_row_digests": len(global_seen),
            "columns": [
                {
                    "index": i,
                    "name": name,
                    "trimmed_name": name.strip(),
                    "dtype_inferred": infer_dtype(
                        global_dtype[i], name.strip().lower() == "label"
                    ),
                    "dtype_evidence_counts": dict(global_dtype[i]),
                    "missing_count": global_missing[i],
                    "positive_infinite_count": global_pos_inf[i],
                    "negative_infinite_count": global_neg_inf[i],
                    "constant": global_constant[i],
                    "constant_value": decode(global_first[i]) if global_constant[i] else None,
                    "majority_candidate": decode(global_majority[i]["candidate"]),
                }
                for i, name in enumerate(reference)
            ],
        },
    }


def infer_dtype(counts: Counter[str], label_col: bool) -> str:
    if label_col:
        return "categorical/string target"
    if counts.get("string", 0):
        return "string"
    if counts.get("float", 0) or counts.get("infinite", 0):
        return "float64-compatible numeric"
    if counts.get("integer", 0):
        return "int64-compatible numeric"
    if counts.get("missing", 0):
        return "all missing/empty"
    return "unknown/no observed values"


def scan_dominant_counts(result: dict[str, Any], paths: list[Path]) -> None:
    overall_candidates = [
        (c["majority_candidate"] or "").encode() for c in result["overall"]["columns"]
    ]
    overall_counts = [0 for _ in overall_candidates]
    for file_stats, path in zip(result["files"], paths):
        candidates = [(c["majority_candidate"] or "").encode() for c in file_stats["columns"]]
        counts = [0 for _ in candidates]
        with path.open("rb") as handle:
            header = split_header(handle.readline())
            col_count = len(header)
            for raw_line in handle:
                row = split_row(raw_line, col_count)
                for idx, cell in enumerate(row):
                    if cell == candidates[idx]:
                        counts[idx] += 1
                    if cell == overall_candidates[idx]:
                        overall_counts[idx] += 1
        file_stats["constant_features"] = []
        file_stats["near_constant_features"] = []
        rows = file_stats["row_count"]
        for idx, column in enumerate(file_stats["columns"]):
            ratio = counts[idx] / rows if rows else 0.0
            column["dominant_value"] = column["majority_candidate"]
            column["dominant_count"] = counts[idx]
            column["dominant_ratio"] = ratio
            column["near_constant"] = ratio >= NEAR_CONSTANT_THRESHOLD
            if column["constant"]:
                file_stats["constant_features"].append(
                    {"index": idx, "name": column["name"], "value": column["constant_value"]}
                )
            elif column["near_constant"]:
                file_stats["near_constant_features"].append(
                    {
                        "index": idx,
                        "name": column["name"],
                        "dominant_value": column["dominant_value"],
                        "dominant_count": counts[idx],
                        "dominant_ratio": ratio,
                    }
                )
        print(f"second pass: {path.name}", flush=True)

    result["overall"]["constant_features"] = []
    result["overall"]["near_constant_features"] = []
    rows = result["overall"]["row_count"]
    for idx, column in enumerate(result["overall"]["columns"]):
        ratio = overall_counts[idx] / rows if rows else 0.0
        column["dominant_value"] = column["majority_candidate"]
        column["dominant_count"] = overall_counts[idx]
        column["dominant_ratio"] = ratio
        column["near_constant"] = ratio >= NEAR_CONSTANT_THRESHOLD
        if column["constant"]:
            result["overall"]["constant_features"].append(
                {"index": idx, "name": column["name"], "value": column["constant_value"]}
            )
        elif column["near_constant"]:
            result["overall"]["near_constant_features"].append(
                {
                    "index": idx,
                    "name": column["name"],
                    "dominant_value": column["dominant_value"],
                    "dominant_count": overall_counts[idx],
                    "dominant_ratio": ratio,
                }
            )


def add_interpretation(result: dict[str, Any]) -> None:
    files_by_day: defaultdict[str, list[str]] = defaultdict(list)
    for file_stats in result["files"]:
        files_by_day[file_stats["day"]].append(file_stats["source_filename"])
    duplicate_names = set(result["duplicate_trimmed_column_names"])
    suspicious = []
    for col in result["overall"]["columns"]:
        name = col["trimmed_name"]
        lower = name.lower()
        reasons = []
        if name == "Destination Port":
            reasons.append("network endpoint/service attribute can encode scenario identity")
        if name in {"Protocol", "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP", "Timestamp"}:
            reasons.append("identifier or temporal metadata can leak across file/time splits")
        if name in duplicate_names:
            reasons.append("duplicate column name appears at multiple positions")
        if col["constant"]:
            reasons.append("constant across all discovered raw files")
        elif col["near_constant"]:
            reasons.append(f"near-constant at {col['dominant_ratio']:.4%}")
        if any(token in lower for token in ("label", "class", "attack")):
            reasons.append("target-like name; keep out of feature matrix")
        if reasons:
            suspicious.append(
                {
                    "index": col["index"],
                    "name": col["name"],
                    "trimmed_name": name,
                    "reasons": sorted(set(reasons)),
                }
            )
    rows = result["overall"]["row_count"]
    feature_count = len(result["overall"]["columns"]) - 1
    dense_bytes = rows * feature_count * 8
    result["interpretation"] = {
        "labels_present_by_file": {
            f["source_filename"]: sorted(f["labels"].keys()) for f in result["files"]
        },
        "attack_types_by_file": {
            f["source_filename"]: sorted(label for label in f["labels"] if label != "BENIGN")
            for f in result["files"]
        },
        "files_by_day": dict(files_by_day),
        "suspicious_identifier_or_leakage_prone_columns": suspicious,
        "memory_requirements": {
            "raw_csv_bytes": result["overall"]["size_bytes"],
            "raw_csv_mb": result["overall"]["size_mb"],
            "dense_float64_feature_matrix_bytes_excluding_label": dense_bytes,
            "dense_float64_feature_matrix_mb_excluding_label": round(dense_bytes / (1024 * 1024), 3),
            "local_resource_risks": [
                "Immediate full concatenation can create multi-GB peak memory usage.",
                "Duplicate-row detection is safer as streaming or disk-backed work.",
                "Missing and infinite values must be handled before estimator training.",
            ],
        },
    }


def write_summary(result: dict[str, Any]) -> None:
    overall = result["overall"]
    binary = overall["binary_class_distribution"]
    inventory = [
        [
            f["source_filename"],
            f["size_mb"],
            f"{f['row_count']:,}",
            f"{f['binary_class_distribution'].get('BENIGN', 0):,}",
            f"{f['binary_class_distribution'].get('ATTACK', 0):,}",
            f"{f['duplicate_rows']:,}",
        ]
        for f in result["files"]
    ]
    labels = [
        [label, f"{count:,}", f"{pct(count, overall['row_count']):.3f}%"]
        for label, count in sorted(overall["labels"].items())
    ]
    labels_by_file = [
        [f["source_filename"], ", ".join(sorted(f["labels"].keys()))]
        for f in result["files"]
    ]
    missing = [
        [c["index"], c["name"], f"{c['missing_count']:,}"]
        for c in overall["columns"]
        if c["missing_count"]
    ]
    infinite = [
        [c["index"], c["name"], f"{c['positive_infinite_count']:,}", f"{c['negative_infinite_count']:,}"]
        for c in overall["columns"]
        if c["positive_infinite_count"] or c["negative_infinite_count"]
    ]
    constants = [[c["index"], c["name"], c["value"]] for c in overall["constant_features"]]
    near_constants = [
        [c["index"], c["name"], c["dominant_value"], f"{c['dominant_ratio']:.4%}"]
        for c in overall["near_constant_features"]
    ]
    suspicious = [
        [c["index"], c["trimmed_name"], "; ".join(c["reasons"])]
        for c in result["interpretation"]["suspicious_identifier_or_leakage_prone_columns"]
    ]

    removals = [
        [c["name"], "constant across all discovered raw files; no predictive variance"]
        for c in overall["constant_features"]
    ]
    removals.extend(
        [
            [c["name"], f"near-constant ({c['dominant_ratio']:.4%} dominant value); review before retention"]
            for c in overall["near_constant_features"]
        ]
    )
    if "Fwd Header Length" in result["duplicate_trimmed_column_names"]:
        removals.append(["one duplicate Fwd Header Length column", "duplicate feature name appears at multiple positions; resolve by position"])
    removals.extend(
        [
            ["Label", "target column; retain as target, exclude from feature matrix"],
            ["Destination Port", "not automatic removal, but service/scenario identity may be leakage-prone"],
        ]
    )
    cleaning = [
        "Trim or canonicalize column names in processed data while preserving a raw-name mapping.",
        "Resolve the duplicate `Fwd Header Length` columns by explicit position.",
        "Convert features to numeric types only after reporting all original labels.",
        "Replace or reject positive infinite values in `Flow Bytes/s` and `Flow Packets/s` by policy.",
        "Handle missing/NaN values in `Flow Bytes/s` before training.",
        "Drop or quarantine exact duplicate rows according to validation policy, not during raw EDA.",
        "Create binary `BENIGN`/`ATTACK` target only after preserving multiclass label inventory.",
    ]
    splits = [
        (
            "Temporal weekday holdout",
            "Train: Monday + Tuesday + Wednesday. Validation: Thursday files. Test: Friday files.",
            "Train covers BENIGN, FTP-Patator, SSH-Patator, DoS family, Heartbleed; validation covers Web Attack/Infiltration; test covers Bot/DDoS/PortScan.",
            "Best temporal isolation, but several attack families are unseen during training.",
        ),
        (
            "Attack-family-aware file holdout",
            "Train: Monday + Tuesday + Wednesday + Thursday morning + Friday DDoS. Validation: Thursday Infiltration + Friday Bot. Test: Friday PortScan.",
            "Train includes most high-volume attacks; validation exercises rare Bot/Infiltration; test isolates PortScan.",
            "Better train-label coverage, weaker temporal purity because Friday is split by file.",
        ),
        (
            "Rare-attack stress holdout",
            "Train: Monday + Tuesday + Wednesday + Friday DDoS/PortScan. Validation: Friday Bot. Test: Thursday Web Attack/Infiltration.",
            "Common attacks are learned; rare web/infiltration labels are held out as stress cases.",
            "Useful stress test, but non-chronological and rare-label metrics may be unstable.",
        ),
    ]

    lines = [
        "# CICIDS2017 Phase 1 EDA Summary",
        "",
        f"Generated from `{result['raw_directory']}` at `{result['generated_at_utc']}` with streaming CSV scans. Raw files were not modified.",
        "",
        "## Dataset Inventory",
        md_table(["File", "MiB", "Rows", "BENIGN", "ATTACK", "Duplicate rows"], inventory),
        "",
        f"Overall: {overall['file_count']} CSV files, {overall['row_count']:,} rows, {overall['size_mb']:.3f} MiB raw CSV, and {overall['duplicate_rows']:,} duplicate rows by exact raw-line digest.",
        "",
        "## Columns and Schema",
        f"All files match the reference header exactly: `{result['column_consistency']['all_files_match_exact_header']}`.",
        f"Column count: {len(result['original_columns'])}. Duplicate original column names: `{result['duplicate_original_column_names']}`. Duplicate trimmed names: `{result['duplicate_trimmed_column_names']}`.",
        f"Columns with leading/trailing whitespace: {len(result['columns_with_leading_or_trailing_whitespace'])}. Exact names and whitespace metadata are in `summary.json`.",
        "",
        "## Labels",
        md_table(["Label", "Rows", "Share"], labels),
        "",
        "## Labels by File",
        md_table(["File", "Labels present"], labels_by_file),
        "",
        "## Binary Distribution",
        f"BENIGN: {binary.get('BENIGN', 0):,} ({pct(binary.get('BENIGN', 0), overall['row_count']):.3f}%). ATTACK: {binary.get('ATTACK', 0):,} ({pct(binary.get('ATTACK', 0), overall['row_count']):.3f}%). This is an EDA grouping only; the production binary mapping is not finalized here.",
        "",
        "## Missing and Infinite Values",
        md_table(["Index", "Column", "Missing count"], missing) if missing else "No missing values were detected under the configured missing-token policy.",
        "",
        md_table(["Index", "Column", "+inf", "-inf"], infinite) if infinite else "No positive or negative infinite values were detected.",
        "",
        "## Constant and Near-Constant Features",
        md_table(["Index", "Column", "Value"], constants) if constants else "No globally constant features were detected.",
        "",
        md_table(["Index", "Column", "Dominant value", "Dominant share"], near_constants) if near_constants else f"No globally near-constant features were detected at {NEAR_CONSTANT_THRESHOLD:.1%}.",
        "",
        "## Suspicious Identifier or Leakage-Prone Columns",
        md_table(["Index", "Column", "Reason"], suspicious) if suspicious else "No obvious identifier columns were present; review `Destination Port` separately.",
        "",
        "## Memory and Local Resource Notes",
        f"Raw CSV size is {overall['size_mb']:.3f} MiB. A dense float64 feature matrix excluding the label would be about {result['interpretation']['memory_requirements']['dense_float64_feature_matrix_mb_excluding_label']:.3f} MiB before dataframe/index/intermediate overhead, so naive full concatenation can require multiple GB of RAM.",
        "",
        "## Recommended Columns to Remove or Review",
        md_table(["Column", "Reason"], removals),
        "",
        "## Explicit Cleaning Operations Required",
    ]
    lines.extend(f"- {item}" for item in cleaning)
    lines.extend(["", "## Possible Day/File-Based Split Strategies"])
    for name, assignment, coverage, tradeoff in splits:
        lines.extend([f"### {name}", assignment, f"Label coverage: {coverage}", f"Trade-off: {tradeoff}", ""])
    lines.extend(
        [
            "## Recommended Split Proposal Requiring Review",
            "Proposed default for the next Phase 1 split implementation: **Temporal weekday holdout**. It best follows the spec preference for temporal/file-based isolation and avoids naive random row-level leakage. The review decision is whether the first baseline should accept stricter unseen-attack-family evaluation or temporarily prioritize broader train-label coverage.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notebook() -> None:
    cells: list[dict[str, Any]] = []

    def md(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})

    def code(source: str) -> None:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)})

    md("# CICIDS2017 Phase 1 Exploratory Data Analysis\n\nThis notebook documents a read-only EDA of the raw CICIDS2017 MachineLearningCSV files under `data/raw/`. It reports data-quality issues without cleaning or changing the raw dataset.")
    md("## Guardrails\n\n- Raw DVC-tracked files are immutable inputs.\n- Original multiclass labels are analyzed before final binary target mapping.\n- Scans are streaming/chunk-friendly and avoid immediate full concatenation.\n- Split discussion is temporal/day/file-based; naive random row-level splits are excluded.")
    code("# Set to True to regenerate reports/eda/summary.json and summary.md from raw CSVs.\n# The full scan is read-only but can take several minutes on a local machine.\nfrom pathlib import Path\nimport runpy\n\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nRUN_FULL_SCAN = False\nif RUN_FULL_SCAN:\n    runpy.run_path(str(ROOT / 'scripts' / 'phase1_eda.py'), run_name='__main__')")
    code("from pathlib import Path\nimport json\n\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nSUMMARY_JSON = ROOT / 'reports' / 'eda' / 'summary.json'\nwith SUMMARY_JSON.open('r', encoding='utf-8') as fh:\n    summary = json.load(fh)\nprint(summary['scan_method'])\nprint(f\"files={summary['overall']['file_count']} rows={summary['overall']['row_count']:,}\")")
    md("## Discovered CSV Files, Sizes, and Row Counts")
    code("for f in summary['files']:\n    print(f\"{f['source_filename']}: {f['size_mb']:.3f} MiB, {f['row_count']:,} rows\")")
    md("## Exact Original Column Names and Whitespace")
    code("for c in summary['original_columns']:\n    ws = ' whitespace' if c['leading_whitespace'] or c['trailing_whitespace'] else ''\n    print(f\"{c['index']:02d}: {c['name']!r} -> {c['trimmed_name']!r}{ws}\")\nprint('duplicate original:', summary['duplicate_original_column_names'])\nprint('duplicate trimmed:', summary['duplicate_trimmed_column_names'])")
    md("## Column Consistency Across Files")
    code("print(json.dumps(summary['column_consistency'], indent=2))")
    md("## Data Types Per Feature")
    code("for c in summary['overall']['columns']:\n    print(f\"{c['index']:02d} {c['name']!r}: {c['dtype_inferred']} {c['dtype_evidence_counts']}\")")
    md("## Labels Present and BENIGN/Attack Distribution")
    code("print('overall labels:', json.dumps(summary['overall']['labels'], indent=2))\nprint('overall binary grouping:', summary['overall']['binary_class_distribution'])\nfor f in summary['files']:\n    print(f\"{f['source_filename']}: labels={f['labels']} binary={f['binary_class_distribution']}\")")
    md("## Missing, Infinite, and Duplicate Values")
    code("for c in summary['overall']['columns']:\n    if c['missing_count'] or c['positive_infinite_count'] or c['negative_infinite_count']:\n        print(c['index'], repr(c['name']), 'missing', c['missing_count'], '+inf', c['positive_infinite_count'], '-inf', c['negative_infinite_count'])\nprint('overall duplicate rows:', summary['overall']['duplicate_rows'])")
    md("## Constant, Near-Constant, Leakage-Prone Columns, and Splits")
    code("print('constant:', summary['overall']['constant_features'])\nprint('near constant:', summary['overall']['near_constant_features'])\nprint('suspicious/review:', json.dumps(summary['interpretation']['suspicious_identifier_or_leakage_prone_columns'], indent=2))\nprint('attack types by file:', json.dumps(summary['interpretation']['attack_types_by_file'], indent=2))\nprint('memory:', json.dumps(summary['interpretation']['memory_requirements'], indent=2))")
    md("## Narrative Report\n\nThe narrative conclusions, required cleaning operations, label coverage trade-offs, and proposed split requiring review are in `reports/eda/summary.md`.")
    NOTEBOOK.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw directory missing: {RAW_DIR}")
    paths = sorted(RAW_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = scan_first_pass(paths)
    scan_dominant_counts(result, paths)
    add_interpretation(result)
    result["scan_total_seconds"] = round(time.time() - started, 3)
    SUMMARY_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(result)
    write_notebook()
    print(f"wrote {rel(NOTEBOOK)}")
    print(f"wrote {rel(SUMMARY_MD)}")
    print(f"wrote {rel(SUMMARY_JSON)}")
    print(f"scan_total_seconds={result['scan_total_seconds']}")


if __name__ == "__main__":
    main()
